#!/usr/bin/env python3
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
import json

import tensorflow as tf
from keras.models import Model
from keras import layers

from .instruction_model import (
    instruction_model_inference,
)


def create_stair_structure(
    features_len: int,
    hidden_sizes: Optional[list[int]] = None,
    use_batch_norm: bool = False,
):
    """
    Creates a stair-structured model graph with an optional batch normalization layer.

    Args:
        features_len: The dimensionality of the input features.
        hidden_sizes: A list of hidden layer sizes. Defaults to [14, 12, 10] if None.
        use_batch_norm: Whether to apply batch normalization on the input.

    Returns:
        A tuple (model_graph, list_of_keras_layers).
    """
    hidden_sizes = hidden_sizes or [14, 12, 10]
    input_layer = InputBuffer(features_len)
    normalizer_layer = NormalizationComputation()
    layers, current_buffer = (
        ([normalizer_layer], normalizer_layer(input_layer))
        if use_batch_norm
        else ([], input_layer)
    )

    buffers = [current_buffer]

    for size in hidden_sizes:
        mid_layer = Dense(size, activation="relu")
        mid_buffer = mid_layer(current_buffer)
        layers.append(mid_layer)
        buffers.append(mid_buffer)
        current_buffer = mid_buffer

    current_buffer = (
        Concatenate()(buffers) if len(buffers) > 1 else current_buffer
    )
    model = ModelGraph(input_layer, current_buffer)

    return model, [layer.keras_layer for layer in layers]


NO_BATCH_NORM = 0
INPLACE = 1
NOT_INPLACE = 2


def ff_model(sizes: list[int], use_batch_norm: int = 0, activations=None):
    """
    Builds a feed-forward model graph based on provided layer sizes and activations.

    Args:
        sizes: A list with structure [features_len, hidden_layer_sizes..., last_layer_size].
        use_batch_norm: Batch normalization mode (NO_BATCH_NORM, INPLACE, or NOT_INPLACE).
        activations: Activation functions for each layer. Can be a list or a shorthand string.

    Returns:
        A ModelGraph representing the feed-forward network.
    """
    features_len, *hidden_sizes, last_layer_size = sizes
    if activations is None:
        activations = ["relu"] * len(hidden_sizes) + ["sigmoid"]
    elif len(activations) != len(hidden_sizes) + 1:
        raise ValueError(
            "The number of activations must match the number of hidden layers + 1."
        )

    if isinstance(activations, str):
        activation_map = {
            "r": "relu",
            "s": "sigmoid",
            "t": "tanh",
            "l": None,
        }
        activations = [activation_map[activation] for activation in activations]

    input_layer = InputBuffer(features_len)
    current_buffer = (
        NormalizationComputation(in_place=use_batch_norm == INPLACE)(input_layer)
        if use_batch_norm != NO_BATCH_NORM
        else input_layer
    )
    for size in hidden_sizes:
        current_buffer = Dense(size, activation=activations.pop(0))(
            current_buffer
        )

    dense = Dense(last_layer_size, activation=activations.pop(0))(
        current_buffer
    )
    model = ModelGraph(input_layer, dense)

    return model


def generate_validation_data(
    features: list[str],
    model: Model,
    means=None,
    stds=None,
):
    """
    Generates validation data by creating random inputs and obtaining the model outputs.

    Args:
        features: List of feature names.
        model: A Keras model instance used for inference.
        means: Optional mean values to add to the inputs.
        stds: Optional standard deviation values to scale the inputs.

    Returns:
        A dictionary with keys "inputs" and "expected_outputs".
    """
    input_data = np.random.randn(10, len(features)).astype(np.float32)

    if stds is not None:
        input_data = input_data * (np.array(stds) + 1e-6)
    if means is not None:
        input_data = input_data + np.array(means)

    output_data = model.predict_on_batch(input_data)

    return {
        "inputs": input_data.tolist(),
        "expected_outputs": output_data.tolist(),
    }


def tau_compare(predictions, y_data):
    """
    Computes Kendall's Tau-b correlation for each output column.

    Args:
        predictions: The model predictions as a NumPy array.
        y_data: The ground truth values as a NumPy array.

    Returns:
        A list of Tau-b scores if multiple columns are present; otherwise a single float.
    """
    n_samples, n_cols = y_data.shape
    results = []

    for col in range(n_cols):
        # Extract the predictions and ground truth for the current column.
        pred_col = predictions[:, col]
        y_col = y_data[:, col]

        tau, p_value = kendalltau(pred_col, y_col)
        if np.isnan(tau):
            tau = 0.0

        results.append(tau)

    return results if len(results) > 1 else results[0]


def score_selection(model, x_data, y_data):
    """
    Computes Kendall's Tau-b for each output column of y_data.

    Args:
        model: A model with a .predict() method or an object processed via instruction_model_inference.
        x_data: Input feature data.
        y_data: Ground truth labels (NumPy array, pd.Series, or pd.DataFrame).

    Returns:
        A list of Tau-b correlation scores, or a single float if y_data has one column.
    """
    if isinstance(y_data, (pd.DataFrame, pd.Series)):
        y_data = y_data.to_numpy()

    if y_data.ndim == 1:
        y_data = y_data.reshape(-1, 1)

    if hasattr(model, "predict"):
        predictions = model.predict(x_data)
    else:
        predictions = instruction_model_inference(model, x_data)[-1]

    if isinstance(predictions, (pd.DataFrame, pd.Series)):
        predictions = predictions.to_numpy()

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    return tau_compare(predictions, y_data)


###############################################################################
# 1. DataBuffer Classes
###############################################################################
class DataBuffer:
    """
    A container for a Keras tensor that also tracks the operation (op) that produced it,
    and maintains a list of input DataBuffers used for that operation.
    """

    def __init__(self, tensor, op=None, inputs=None):
        self.tensor = tensor  # The underlying Keras (symbolic) tensor.
        self.op = op  # The ComputationOp that produced this buffer (if any).
        self.inputs = inputs if inputs is not None else []  # Always a list.

    def __repr__(self):
        return f"DataBuffer(shape={self.tensor.shape})"

    @property
    def os(self):
        """
        Shortcut for output size.
        Assumes the tensor shape is (None, output_size) and returns that output_size.
        """
        return int(self.tensor.shape[-1])

    def __getitem__(self, key):
        """
        Overloads indexing for DataBuffer. When a list, tuple, slice, or np.ndarray is
        provided as key, it automatically creates a CopyMaskedComputation layer to select
        the corresponding columns.

        Example:
            input_buffer = InputBuffer(3)
            sliced = input_buffer[[2, 0]]
        """
        if isinstance(key, int):
            indexes = [key]
        elif isinstance(key, slice):
            indexes = list(range(self.os))[key]
        elif isinstance(key, (list, tuple, np.ndarray)):
            indexes = list(key)
        else:
            raise TypeError("Unsupported index type for DataBuffer: " + str(type(key)))
        return CopyMaskedComputation(indexes)([self])


class InputBuffer(DataBuffer):
    """
    Wraps a Keras Input. Can be instantiated with an integer (e.g. InputBuffer(15))
    or a tuple (e.g. InputBuffer((15,))).
    """

    def __init__(self, shape_or_os, name=None):
        if isinstance(shape_or_os, int):
            shape = (shape_or_os,)
        else:
            shape = shape_or_os
        inp = layers.Input(shape=shape, name=name)
        super().__init__(inp, op=None, inputs=[])
        self.shape = shape  # MODIFIED: Store the input shape explicitly.

    def __repr__(self):
        return f"InputBuffer(shape={self.shape})"

    def __call__(self, *args, **kwargs):
        return self


###############################################################################
# 2. ComputationOp Base Class and Existing Ops
###############################################################################
class ComputationOp(ABC):
    """
    Base class for operations that wrap Keras layers.
    Every op's __call__ expects a list of DataBuffer objects.
    """

    def __init__(self):
        self.keras_layer = None

    @abstractmethod  # MODIFIED: Mark __call__ as an abstract method.
    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        raise NotImplementedError("Subclasses must implement __call__.")

    @abstractmethod  # MODIFIED: Mark compile_instructions as an abstract method.
    def compile_instructions(
        self, input_indices, weights_visited, model_structure
    ) -> int:
        raise NotImplementedError("Subclasses must implement compile_instructions.")


class ActivationComputation(ComputationOp):
    """
    An activation operation that applies an activation function elementwise.
    For "RELU", "SIGMOID", "TANH", "SOFTMAX", the native Keras Activation layer is used.
    For custom activations (e.g., "SQRT", "LOG", "LOG10", "INVERSE"), a layers.Lambda layer is created.

    The compile_instructions method creates an in-place activation instruction.
    """

    def __init__(self, activation, in_place=True, name: Optional[str] = None):
        super().__init__()
        self.in_place = in_place
        self.activation = activation.upper()
        self.name = name
        if self.activation in {"RELU", "SIGMOID", "TANH", "SOFTMAX"}:
            self.keras_layer = tf.keras.layers.Activation(
                self.activation.lower(), name=name
            )
        elif self.activation == "SQRT":
            self.keras_layer = layers.Lambda(
                lambda x: tf.where(x > 0, tf.sqrt(x), tf.zeros_like(x)), name=name
            )
        elif self.activation == "LOG":
            self.keras_layer = layers.Lambda(
                lambda x: tf.math.log(tf.maximum(x, 0) + 1), name=name
            )
        elif self.activation == "LOG10":
            self.keras_layer = layers.Lambda(
                lambda x: tf.math.log(tf.maximum(x, 0) + 1) / tf.math.log(10.0),
                name=name,
            )
        elif self.activation == "INVERSE":
            self.keras_layer = layers.Lambda(lambda x: 1 - x, name=name)
        else:
            raise ValueError(f"Unexpected activation: {self.activation}")

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        y = self.keras_layer(inputs[0].tensor)
        return DataBuffer(y, op=self, inputs=inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        if len(input_indices) != 1:
            raise ValueError("ActivationComputation expects exactly one input.")
        # The activation operation is in-place in the instruction model.

        if self.in_place:
            target_index = input_indices[0]
        else:
            output_index = len(model_structure["buffer_sizes"])
            model_structure["buffer_sizes"].append(
                model_structure["buffer_sizes"][input_indices[0]]
            )
            copy_instr = {
                "type": "COPY",
                "input": input_indices[0],
                "output": output_index,
                "internal_index": 0,
            }
            model_structure["instructions"].append(copy_instr)
            target_index = output_index

        instr = {
            "type": "ACTIVATION",
            "input": target_index,
            "activation": self.activation,
        }
        model_structure["instructions"].append(instr)
        return target_index


class Dense(ComputationOp):
    """
    A dense operation that expects its input as a one-element list.
    Shared weights are stored so that repeated calls reuse the same weight index.
    """

    def __init__(self, output_size, activation=None, name=None):
        super().__init__()
        self.input_size = None
        self.output_size = output_size
        self.activation = activation
        self.keras_layer = layers.Dense(output_size, activation=activation, name=name)

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("Dense expects exactly one input in the list.")
        input_tensor = inputs[0].tensor
        output_tensor = self.keras_layer(input_tensor)
        if self.input_size is None:
            self.input_size = int(input_tensor.shape[-1])
        return DataBuffer(output_tensor, op=self, inputs=inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        if len(input_indices) != 1:
            raise ValueError(
                "Dense.compile_instructions expects one input index."
            )
        wv = weights_visited["weights"]
        if id(self) not in wv:
            wv[id(self)] = len(model_structure["weights"])
            weights, bias = self.keras_layer.get_weights()
            model_structure["weights"].append(
                weights.T.tolist()
            )  # stored as (output, input)
            model_structure["bias"].append(bias.tolist())

        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(self.output_size)
        instr = {
            "type": "DOT",
            "input": input_indices[0],
            "output": output_index,
            "weights": wv[id(self)],
        }
        if self.activation is not None:
            instr["activation"] = self.activation.upper()
        model_structure["instructions"].append(instr)
        return output_index


class CopyMaskedComputation(ComputationOp):
    """
    A copy-masked operation that selects specific columns from the input.
    """

    def __init__(self, indexes, name=None):
        super().__init__()
        self.indexes = indexes  # List of column indices to select.
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        indices_tensor = tf.constant(self.indexes, dtype=tf.int32)
        gather_layer = layers.Lambda(lambda x: tf.gather(x, indices=indices_tensor, axis=1))
        output_tensor = gather_layer(inputs[0].tensor)
        return DataBuffer(output_tensor, op=self, inputs=inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(len(self.indexes))
        instr = {
            "type": "COPY_MASKED",
            "input": input_indices[0],
            "output": output_index,
            "indexes": self.indexes,
        }
        model_structure["instructions"].append(instr)

        return output_index


class Concatenate(ComputationOp):
    """
    A concatenation operation that takes a list of inputs and concatenates them along the last axis.
    """

    def __init__(self, axis=-1, name=None):
        super().__init__()
        self.axis = axis
        self.keras_layer = layers.Concatenate(axis=axis, name=name)

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        tensors = [inp.tensor for inp in inputs]
        output_tensor = self.keras_layer(tensors)
        return DataBuffer(output_tensor, op=self, inputs=inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        offsets = []
        total_size = 0
        for idx in input_indices:
            offsets.append(total_size)
            total_size += model_structure["buffer_sizes"][idx]

        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(total_size)

        for idx, offset in zip(input_indices, offsets):
            instr = {
                "type": "COPY",
                "input": idx,
                "output": output_index,
                "internal_index": offset,
            }
            model_structure["instructions"].append(instr)
        return output_index


class NormalizationComputation(ComputationOp):
    """
    A normalization operation that wraps a BatchNormalization layer.
    """

    def __init__(self, in_place=True, center=True, scale=True, epsilon=1e-3, name=None):
        super().__init__()
        self.in_place = in_place
        self.epsilon = epsilon
        self.keras_layer = tf.keras.layers.BatchNormalization(
            epsilon=epsilon, center=center, scale=scale, axis=-1, name=name
        )

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        new_tensor = self.keras_layer(inputs[0].tensor)
        return DataBuffer(new_tensor, op=self, inputs=inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        weights = self.keras_layer.get_weights()

        if len(weights) == 4:
            gamma, beta, moving_mean, moving_variance = weights
        elif len(weights) == 3:
            if self.keras_layer.center and not self.keras_layer.scale:
                beta, moving_mean, moving_variance = weights
                gamma = np.ones_like(moving_mean)
            else:
                gamma, moving_mean, moving_variance = weights
                beta = np.zeros_like(gamma)
        elif len(weights) == 2:
            moving_mean, moving_variance = weights
            gamma = np.ones_like(moving_mean)
            beta = np.zeros_like(moving_mean)
        else:
            raise ValueError(
                f"Unexpected number of BN weights returned: {len(weights)}. "
                "Check your 'center' and 'scale' arguments."
            )

        epsilon = self.epsilon
        std = gamma / np.sqrt(moving_variance + epsilon)
        center = -moving_mean

        pw = weights_visited["parameters"]
        if self.in_place:
            target_index = input_indices[0]
        else:
            output_index = len(model_structure["buffer_sizes"])
            model_structure["buffer_sizes"].append(
                model_structure["buffer_sizes"][input_indices[0]]
            )
            copy_instr = {
                "type": "COPY",
                "input": input_indices[0],
                "output": output_index,
                "internal_index": 0,
            }
            model_structure["instructions"].append(copy_instr)
            target_index = output_index

        if id(self) not in pw:
            pw[id(self)] = [len(model_structure["parameters"]) + i for i in range(3)]
            model_structure["parameters"].append(center.tolist())
            model_structure["parameters"].append(std.tolist())
            model_structure["parameters"].append(beta.tolist())

        instr_center = {
            "type": "ADD_ELEMENTWISE",
            "input": target_index,
            "parameters": pw[id(self)][0],
        }
        instr_mul = {
            "type": "MUL_ELEMENTWISE",
            "input": target_index,
            "parameters": pw[id(self)][1],
        }
        instr_add = {
            "type": "ADD_ELEMENTWISE",
            "input": target_index,
            "parameters": pw[id(self)][2],
        }

        model_structure["instructions"].append(instr_center)
        model_structure["instructions"].append(instr_mul)
        model_structure["instructions"].append(instr_add)

        return target_index


class Attention(ComputationOp):
    """
    An attention operation that computes softmax attention from a key buffer and applies it
    elementwise to a target buffer.
    """

    def __init__(self, name=None):
        super().__init__()
        self.a = None  # Target dimension (and output dimension)
        self.b = None  # Key dimension
        self.keras_layer = None
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("Attention expects two inputs: [target, key]")
        target, key = inputs

        if self.keras_layer is None:
            self.a = target.os
            self.b = key.os
            self.keras_layer = layers.Dense(self.a, name=self.name, activation="softmax")

        softmaxed = self.keras_layer(key.tensor)
        result_tensor = target.tensor * softmaxed
        return DataBuffer(result_tensor, op=self, inputs=inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        wv = weights_visited["weights"]
        if id(self) not in wv:
            wv[id(self)] = len(model_structure["weights"])
            weights, bias = self.keras_layer.get_weights()
            model_structure["weights"].append(weights.T.tolist())
            model_structure["bias"].append(bias.tolist())

        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(self.a)

        instr = {
            "type": "ATTENTION",
            "input": input_indices[0],
            "key": input_indices[1],
            "output": output_index,
            "weights": wv[id(self)],
        }
        model_structure["instructions"].append(instr)
        return output_index


class Add(ComputationOp):
    """
    An elementwise addition operation that adds a list of input DataBuffers.
    Internally it uses keras.layers.Add.
    For instruction purposes, the instruction type is "ADD_ELEMENTWISE_BUFFERS"
    and the input field is a list of buffer indices.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.keras_layer = tf.keras.layers.Add(name=name)

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        # Gather the underlying tensors from each DataBuffer.
        tensors = [inp.tensor for inp in inputs]
        output_tensor = self.keras_layer(tensors)
        return DataBuffer(output_tensor, op=self, inputs=inputs)

    def compile_instructions(
        self, input_indices, weights_visited, model_structure
    ) -> int:
        if not input_indices:
            raise ValueError("Add expects at least one input.")
        # Assume all input buffers have the same output size.
        out_size = model_structure["buffer_sizes"][input_indices[0]]
        for idx in input_indices:
            if model_structure["buffer_sizes"][idx] != out_size:
                raise ValueError("All inputs must have the same size for addition.")
        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(out_size)
        instr = {
            "type": "ADD_ELEMENTWISE_BUFFERS",
            "input": input_indices,  # Note: the entire list of input indices is provided.
            "output": output_index,
        }
        model_structure["instructions"].append(instr)
        return output_index


class Multiply(ComputationOp):
    """
    An elementwise multiplication operation that multiplies a list of input DataBuffers.
    Internally it uses keras.layers.Multiply.
    For instruction purposes, the instruction type is "MULTIPLY_ELEMENTWISE_BUFFERS"
    and the input field is a list of buffer indices.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.keras_layer = tf.keras.layers.Multiply(name=name)

    def __call__(self, inputs):
        if not isinstance(inputs, list) or len(inputs) < 2:
            raise ValueError("Multiply expects at least two inputs.")
        tensors = [inp.tensor for inp in inputs]
        output_tensor = self.keras_layer(tensors)
        return DataBuffer(output_tensor, op=self, inputs=inputs)

    def compile_instructions(
        self, input_indices, weights_visited, model_structure
    ) -> int:
        if not input_indices:
            raise ValueError("Multiply expects at least one input.")
        out_size = model_structure["buffer_sizes"][input_indices[0]]
        for idx in input_indices:
            if model_structure["buffer_sizes"][idx] != out_size:
                raise ValueError(
                    "All inputs must have the same size for multiplication."
                )
        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(out_size)
        instr = {
            "type": "MULTIPLY_ELEMENTWISE_BUFFERS",
            "input": input_indices,
            "output": output_index,
        }
        model_structure["instructions"].append(instr)
        return output_index


###############################################################################
# 3. ModelGraph and Instruction Model Compilation
###############################################################################
class ModelGraph(ComputationOp):
    """
    Holds the connection between an array of input DataBuffers and an output DataBuffer,
    along with references to all internal Keras layers (for training).
    """

    def __init__(self, input_buffers, output_buffer: DataBuffer, name=None):
        super().__init__()
        if not isinstance(input_buffers, list):
            input_buffers = [input_buffers]
        self.input_buffers = input_buffers
        self.output_buffer = output_buffer
        self._keras_model = Model(
            inputs=[buf.tensor for buf in self.input_buffers],
            outputs=self.output_buffer.tensor,
            name=name,
        )

    @property
    def os(self):
        """
        Shortcut for output size.
        """
        return self.output_buffer.os

    def get_keras(self):
        return self._keras_model

    def compile(self, *args, **kwargs):
        return self._keras_model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self._keras_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._keras_model.predict(*args, **kwargs)

    def predict_on_batch(self, *args, **kwargs):
        return self._keras_model.predict_on_batch(*args, **kwargs)

    def summary(self):
        return self._keras_model.summary()

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        out_tensor = self._keras_model([inp.tensor for inp in inputs])
        return DataBuffer(out_tensor, op=self, inputs=inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        visited = {}

        def traverse(buffer: DataBuffer):
            if id(buffer) in visited:
                return visited[id(buffer)]
            if isinstance(buffer, InputBuffer):
                index = self.input_buffers.index(buffer)
                idx = input_indices[index]
                visited[id(buffer)] = idx
                return idx
            elif buffer.op is None:
                idx = len(model_structure["buffer_sizes"])
                model_structure["buffer_sizes"].append(buffer.os)
                visited[id(buffer)] = idx
                return idx
            else:
                input_idx = [traverse(inp) for inp in buffer.inputs]

                idx = buffer.op.compile_instructions(
                    input_idx, weights_visited, model_structure
                )
                visited[id(buffer)] = idx
                return idx

        traverse(self.output_buffer)
        return visited[id(self.output_buffer)]

    def create_instruction_model(self, weights_visited=None, features=None):
        model_structure = {
            "features": features or [],
            "buffer_sizes": [],
            "instructions": [],
            "maps": [],
            "weights": [],
            "bias": [],
            "parameters": [],
        }

        if weights_visited is None:
            weights_visited = {
                "weights": {},
                "parameters": {},
                "maps": {},
            }

        visited = {}
        for input_buffer in self.input_buffers:
            if id(input_buffer) not in visited:
                idx = len(model_structure["buffer_sizes"])
                model_structure["buffer_sizes"].append(
                    int(input_buffer.tensor.shape[-1])
                )
                visited[id(input_buffer)] = idx

        def traverse(buffer: DataBuffer):
            if id(buffer) in visited:
                return visited[id(buffer)]
            if buffer.op is None:
                idx = len(model_structure["buffer_sizes"])
                model_structure["buffer_sizes"].append(int(buffer.tensor.shape[-1]))
                visited[id(buffer)] = idx
                return idx
            else:
                input_indices = [traverse(inp) for inp in buffer.inputs]
                idx = buffer.op.compile_instructions(
                    input_indices, weights_visited, model_structure
                )
                visited[id(buffer)] = idx
                return idx

        traverse(self.output_buffer)

        for input_buffer in self.input_buffers:
            if id(input_buffer) not in visited:
                raise ValueError(f"Input buffer {input_buffer} was not visited.")

        return model_structure


def create_model_graph(inputs, output: DataBuffer) -> ModelGraph:
    if not isinstance(inputs, list):
        inputs = [inputs]
    return ModelGraph(inputs, output)


def create_instruction_model(inputs, output: DataBuffer):
    return create_model_graph(inputs, output).create_instruction_model()


def validate_keras_model(keras_model, validation_data):
    """
    Validates the Keras model using provided validation data.
    """
    x_val = np.array(validation_data["inputs"])
    y_expected = np.array(validation_data["expected_outputs"])
    y_pred = keras_model.predict(x_val)
    if np.allclose(y_expected, y_pred, atol=1e-6):
        print("Keras model validation successful: predictions match expected outputs.")
    else:
        print("Keras model validation failed.")
        print("Expected outputs:", y_expected)
        print("Predictions:", y_pred)
        raise AssertionError("Keras model validation failed.")
