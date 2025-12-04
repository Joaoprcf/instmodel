#!/usr/bin/env python3
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
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

    current_buffer = Concatenate()(buffers) if len(buffers) > 1 else current_buffer
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
        current_buffer = Dense(size, activation=activations.pop(0))(current_buffer)

    dense = Dense(last_layer_size, activation=activations.pop(0))(current_buffer)
    model = ModelGraph(input_layer, dense)

    return model


import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Optional


class MultiOneHotDenseEncoder(tf.keras.layers.Layer):
    """Learn a dense *embedding* for a sparse set of integer IDs.

    During the forward pass it:
        1. Maps raw integer IDs (possibly non-contiguous) → bucket indices.
        2. One-hot encodes those indices (depth = N+1 where the last bucket is
           unknown / out-of-vocabulary).
        3. Applies a single bias-less `Dense` layer that projects the one-hot
           vector to `output_dim`.

    Parameters
    ----------
    train_ids : List[int]
        All IDs encountered in the training data. They need **not** be
        contiguous and may include negatives. They *must* be hashable.
    output_dim : int
        Dimensionality of the learned vector.
    default_id : int, optional (default = -1)
        The ID that represents the unknown / OOV token. If this ID is *not*
        in `train_ids` it will automatically be treated as OOV.
    key_dtype : tf.DType, optional (default = tf.int64)
        Integer dtype for the lookup table.
    """

    def __init__(
        self,
        feature_indexes: List[int],
        training_ids: List[List[int]],
        output_dims: List[int],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.feature_indexes = feature_indexes
        self.training_ids = training_ids
        self.output_dims = output_dims

        self.singleIdEncoders = [
            OneHotDenseEncoder(
                train_ids=train_ids,
                output_dim=output_dim,
            )
            for train_ids, output_dim in zip(training_ids, output_dims)
        ]

        self.added_buffer_size = sum(output_dims) - len(feature_indexes)

    def call(self, inputs):
        # Input validation
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("Dense expects exactly one input in the list.")

        input_tensor = inputs[0]

        input_size = int(input_tensor.shape[1])

        indexes_to_keep = [
            i for i in range(input_size) if i not in self.feature_indexes
        ]

        indices_tensor = tf.constant(indexes_to_keep, dtype=tf.int32)

        gather_layer_output = layers.Lambda(
            lambda x: tf.gather(x, indices_tensor, axis=1),
            name="gather_keep_other_features",
        )(input_tensor)

        # Apply the singleIdEncoders to the selected features
        encoded_features = [
            encoder(input_tensor[:, index])
            for index, encoder in zip(self.feature_indexes, self.singleIdEncoders)
        ]

        concatenated = layers.Concatenate()([gather_layer_output] + encoded_features)

        return concatenated


class OneHotDenseEncoder(tf.keras.layers.Layer):
    """Learn a dense *embedding* for a sparse set of integer IDs.

    During the forward pass it:
        1. Maps raw integer IDs (possibly non-contiguous) → bucket indices.
        2. One-hot encodes those indices (depth = N+1 where the last bucket is
           unknown / out-of-vocabulary).
        3. Applies a single bias-less `Dense` layer that projects the one-hot
           vector to `output_dim`.

    Parameters
    ----------
    train_ids : List[int]
        All IDs encountered in the training data. They need **not** be
        contiguous and may include negatives. They *must* be hashable.
    output_dim : int
        Dimensionality of the learned vector.
    default_id : int, optional (default = -1)
        The ID that represents the unknown / OOV token. If this ID is *not*
        in `train_ids` it will automatically be treated as OOV.
    key_dtype : tf.DType, optional (default = tf.int64)
        Integer dtype for the lookup table.
    """

    def __init__(
        self,
        train_ids: List[int],
        output_dim: int,
        default_id: int = -1,
        key_dtype: tf.DType = tf.int64,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.key_dtype = key_dtype
        self.default_id = int(default_id)

        # Deduplicate & preserve order
        seen_ids = []
        seen_set = set()
        for _id in train_ids:
            i = int(_id)
            if i in seen_set:
                raise ValueError(f"Duplicate ID found in train_ids: {_id}")
            seen_ids.append(i)
            seen_set.add(i)

        if self.default_id in seen_set:
            raise ValueError(f"default_id ({self.default_id}) must not be in train_ids")

        # Python int count for static shapes & default_value
        num_seen = len(seen_ids)  # N
        self.depth = num_seen + 1  # one extra for unknown
        self.oov_bucket = num_seen  # OOV maps to last bucket

        # Build lookup tensor: maps ID -> bucket index
        # We create a tensor where index i contains the bucket for ID i
        # For IDs not in train_ids, we use oov_bucket
        self._seen_ids = seen_ids
        self._id_to_bucket = {int(id_): bucket for bucket, id_ in enumerate(seen_ids)}

        # Dense projection (no bias) → embedding vector
        self.dense = layers.Dense(output_dim, use_bias=False, dtype=tf.float32)

    def call(self, raw_ids):
        """
        raw_ids: tf.Tensor of floats (e.g. float32) containing integer IDs
        (including the default_id) which will be rounded → cast → looked up.
        """
        # 1) Round & cast floats → integer keys
        int_ids = tf.cast(tf.math.round(raw_ids), self.key_dtype)
        int_ids = tf.reshape(int_ids, [-1])

        # 2) Map IDs to bucket indices using vectorized lookup
        # Use tf.where with conditions for each known ID
        bucket_idx = tf.fill(tf.shape(int_ids), tf.constant(self.oov_bucket, dtype=self.key_dtype))
        for id_, bucket in self._id_to_bucket.items():
            bucket_idx = tf.where(
                tf.equal(int_ids, tf.constant(id_, dtype=self.key_dtype)),
                tf.constant(bucket, dtype=self.key_dtype),
                bucket_idx
            )

        # 3) One-hot encode → (..., depth)
        oh = tf.one_hot(bucket_idx, depth=self.depth, dtype=tf.float32)

        # 4) Project → (..., output_dim)
        return self.dense(oh)

    @property
    def os(self):
        """
        Shortcut for output size.
        Assumes the tensor shape is (None, output_size) and returns that output_size.
        """
        return int(self.dense.shape[-1])

    @property
    def weight_matrix(self) -> tf.Tensor:
        """Returns the (depth x output_dim) weight matrix of the Dense layer."""
        return self.dense.kernel


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

    def __init__(self, activation, in_place=False, name: Optional[str] = None):
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


class ReduceSum(ComputationOp):
    """
    Sums all elements of the input buffer, producing a single output value.
    No trainable parameters.

    Input: (batch, N) → Output: (batch, 1)
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.keras_layer = layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=-1, keepdims=True),
            name=name,
        )

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("ReduceSum expects exactly one input.")
        output_tensor = self.keras_layer(inputs[0].tensor)
        return DataBuffer(output_tensor, op=self, inputs=inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        if len(input_indices) != 1:
            raise ValueError("ReduceSum expects exactly one input.")

        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(1)

        instr = {
            "type": "REDUCE_SUM",
            "input": input_indices[0],
            "output": output_index,
        }
        model_structure["instructions"].append(instr)
        return output_index


class Dense(ComputationOp):
    """
    A dense operation that expects its input as a one-element list.
    Shared weights are stored so that repeated calls reuse the same weight index.

    Parameters
    ----------
    output_size : int
        Number of units in the dense layer.
    activation : str or None, default None
        Activation applied after the matrix multiply.
    use_bias : bool, default True
        Whether to include a bias term.  When False, the compiler stores the
        string `"all 0s"` in the bias slot so downstream code can treat it as
        a zero-vector of the correct shape.
    name : str or None, default None
        Name forwarded to the underlying Keras layer.
    """

    def __init__(self, output_size, activation=None, use_bias=True, name=None):
        super().__init__()
        self.input_size = None
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        self.keras_layer = layers.Dense(
            output_size,
            activation=activation,
            use_bias=use_bias,
            name=name,
        )

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
            raise ValueError("Dense.compile_instructions expects one input index.")

        wv = weights_visited["weights"]
        if id(self) not in wv:  # first time we see this op
            wv[id(self)] = len(model_structure["weights"])

            keras_weights = self.keras_layer.get_weights()
            kernel = keras_weights[0]  # (in, out)
            model_structure["weights"].append(kernel.T.tolist())

            if self.use_bias:
                bias = keras_weights[1]
            else:
                # real zero-vector the same length as output_size
                bias = [0.0] * self.output_size
            model_structure["bias"].append(bias)

        # allocate output buffer
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
        gather_layer = layers.Lambda(
            lambda x: tf.gather(x, indices=indices_tensor, axis=1)
        )
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


class MultiIdEmbeddings(ComputationOp):
    """`ComputationOp` wrapper around `MultiOneHotDenseEncoder`.

    * **Training**: behaves like the `tf.keras.layers.Layer` above, allowing the
      graph to be trained end-to-end with back-prop + optimiser.
    * **Compilation**: converts learned parameters to a constant lookup table
      (`model["maps"]`) + a `MAP_TRANSFORM` instruction.

    This keeps the *runtime* dependency-free while letting you train with the
    full TF/Keras stack.
    """

    def __init__(
        self,
        feature_indexes: List[int],
        training_ids: List[List[int]],
        output_dims: List[int],
        name: Optional[str] = None,
    ):
        super().__init__()
        self.layer = MultiOneHotDenseEncoder(
            feature_indexes=feature_indexes,
            training_ids=training_ids,
            output_dims=output_dims,
        )

        self.feature_indexes = feature_indexes
        self.training_ids = training_ids
        self.output_dims = output_dims

        self.name = name

    # ------------------------------------------------------------------
    # Forward / eager execution
    # ------------------------------------------------------------------
    def __call__(self, inputs):
        # Ensure list wrapping
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("MultiIdEmbeddings expects exactly one input buffer.")

        out_tensor = self.layer(inputs[0].tensor)
        return DataBuffer(out_tensor, op=self, inputs=inputs)

    # ------------------------------------------------------------------
    # Compilation to instruction model
    # ------------------------------------------------------------------
    def compile_instructions(
        self,
        input_indices: List[int],
        weights_visited: Dict[str, Dict[int, Any]],
        model_structure: Dict[str, Any],
    ) -> int:
        if len(input_indices) != 1:
            raise ValueError(
                "MultiIdEmbeddings.compile_instructions expects one input index."
            )

        output_index = len(model_structure["buffer_sizes"])

        output_size = (
            model_structure["buffer_sizes"][input_indices[0]]
            + self.layer.added_buffer_size
        )

        model_structure["buffer_sizes"].append(output_size)

        indexes_to_keep = [
            i
            for i in range(model_structure["buffer_sizes"][input_indices[0]])
            if i not in self.feature_indexes
        ]

        model_structure["instructions"].append(
            {
                "type": "COPY_MASKED",
                "input": input_indices[0],
                "output": output_index,
                "indexes": indexes_to_keep,
            }
        )

        internal_output_index = len(indexes_to_keep)

        # Cache / build `maps` entry only once per layer instance
        maps_cache = weights_visited["maps"]
        # if id(self) not in maps_cache:
        # Convert TF weight matrix to NumPy → Python list

        for encoder_id, encoder in enumerate(self.layer.singleIdEncoders):
            weight_matrix = encoder.weight_matrix.numpy()  # (depth, output_dim)
            map_dict: Dict[int, List[float]] = {}

            # Map each *seen* ID to its learned vector (bucket index = its row)
            for bucket_idx, real_id in enumerate(self.training_ids[encoder_id]):
                vector = weight_matrix[bucket_idx].astype(np.float32).tolist()
                map_dict[int(real_id)] = vector

            # Default / OOV bucket is the *last* row (index = N)
            default_vector = weight_matrix[-1].astype(np.float32).tolist()
            maps_cache[id(self)] = len(model_structure["maps"])
            model_structure["maps"].append(map_dict)

            map_index = maps_cache[id(self)]

            # Emit MAP_TRANSFORM: unknown IDs map → default_vector (row N)
            model_structure["instructions"].append(
                {
                    "type": "MAP_TRANSFORM",
                    "input": input_indices[0],
                    "output": output_index,
                    "internal_input_index": self.feature_indexes[encoder_id],
                    "internal_output_index": internal_output_index,
                    "map": map_index,
                    "size": self.output_dims[encoder_id],
                    "default": default_vector,
                }
            )

            internal_output_index += self.output_dims[encoder_id]

        return output_index


###############################################################################
# ComputationOp wrapper that compiles to a single MAP_TRANSFORM instruction    #
###############################################################################


class SingleIdEmbeddings(ComputationOp):
    """`ComputationOp` wrapper around `OneHotDenseEncoder`.

    * **Training**: behaves like the `tf.keras.layers.Layer` above, allowing the
      graph to be trained end-to-end with back-prop + optimiser.
    * **Compilation**: converts learned parameters to a constant lookup table
      (`model["maps"]`) + a `MAP_TRANSFORM` instruction.

    This keeps the *runtime* dependency-free while letting you train with the
    full TF/Keras stack.
    """

    def __init__(
        self,
        train_ids: List[int],
        output_dim: int,
        internal_input_index: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__()
        default_id = -1
        self.layer = OneHotDenseEncoder(
            train_ids=train_ids, output_dim=output_dim, default_id=default_id
        )
        self.output_dim = int(output_dim)
        self.default_id = int(default_id)
        self.train_ids = list(dict.fromkeys(train_ids))  # deduplicated, ordered
        self.internal_input_index = internal_input_index
        self.name = name

    # ------------------------------------------------------------------
    # Forward / eager execution
    # ------------------------------------------------------------------
    def __call__(self, inputs):
        # Ensure list wrapping
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("SingleIdEmbeddings expects exactly one input buffer.")

        out_tensor = self.layer(inputs[0].tensor)
        return DataBuffer(out_tensor, op=self, inputs=inputs)

    # ------------------------------------------------------------------
    # Compilation to instruction model
    # ------------------------------------------------------------------
    def compile_instructions(
        self,
        input_indices: List[int],
        weights_visited: Dict[str, Dict[int, Any]],
        model_structure: Dict[str, Any],
    ) -> int:
        if len(input_indices) != 1:
            raise ValueError(
                "SingleIdEmbeddings.compile_instructions expects one input index."
            )

        # Cache / build `maps` entry only once per layer instance
        maps_cache = weights_visited["maps"]
        # if id(self) not in maps_cache:
        # Convert TF weight matrix to NumPy → Python list
        weight_matrix = self.layer.weight_matrix.numpy()  # (depth, output_dim)
        map_dict: Dict[int, List[float]] = {}

        # Map each *seen* ID to its learned vector (bucket index = its row)
        for bucket_idx, real_id in enumerate(self.train_ids):
            vector = weight_matrix[bucket_idx].astype(np.float32).tolist()
            map_dict[int(real_id)] = vector

        # Default / OOV bucket is the *last* row (index = N)
        default_vector = weight_matrix[-1].astype(np.float32).tolist()
        maps_cache[id(self)] = len(model_structure["maps"])
        model_structure["maps"].append(map_dict)

        # else:
        #     default_vector = model_structure["maps"][maps_cache[id(self)]][
        #         0
        #     ]  # dummy fetch; we overwrite below
        #     print("BIG PROBLEM??")

        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(self.output_dim)

        map_index = maps_cache[id(self)]

        # Emit MAP_TRANSFORM: unknown IDs map → default_vector (row N)
        model_structure["instructions"].append(
            {
                "type": "MAP_TRANSFORM",
                "input": input_indices[0],
                "output": output_index,
                "internal_input_index": self.internal_input_index,
                "internal_output_index": 0,
                "map": map_index,
                "size": self.output_dim,
                "default": default_vector,
            }
        )

        return output_index


class NormalizationComputation(ComputationOp):
    """
    A normalization operation that wraps a BatchNormalization layer.
    """

    def __init__(self, in_place=False, center=True, scale=True, epsilon=1e-3, name=None):
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


class ScaleVectorized(ComputationOp):
    """
    Multiplies a buffer elementwise by a fixed vector.
    Compiles to a MUL_ELEMENTWISE instruction.
    """

    def __init__(self, scaling_vector, in_place=False, name=None):
        super().__init__()
        self.scaling_vector = np.asarray(scaling_vector, dtype=np.float32)
        self.in_place = in_place
        self.name = name
        self._scaling_tensor = tf.constant(self.scaling_vector, dtype=tf.float32)
        self.keras_layer = layers.Lambda(
            lambda x: x * self._scaling_tensor, name=name
        )

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("ScaleVectorized expects exactly one input.")
        output_tensor = self.keras_layer(inputs[0].tensor)
        return DataBuffer(output_tensor, op=self, inputs=inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        if len(input_indices) != 1:
            raise ValueError("ScaleVectorized expects exactly one input.")

        pw = weights_visited["parameters"]

        if self.in_place:
            target_index = input_indices[0]
        else:
            target_index = len(model_structure["buffer_sizes"])
            model_structure["buffer_sizes"].append(
                model_structure["buffer_sizes"][input_indices[0]]
            )
            copy_instr = {
                "type": "COPY",
                "input": input_indices[0],
                "output": target_index,
                "internal_index": 0,
            }
            model_structure["instructions"].append(copy_instr)

        if id(self) not in pw:
            pw[id(self)] = len(model_structure["parameters"])
            model_structure["parameters"].append(self.scaling_vector.tolist())

        instr = {
            "type": "MUL_ELEMENTWISE",
            "input": target_index,
            "parameters": pw[id(self)],
        }
        model_structure["instructions"].append(instr)
        return target_index


class ShiftVectorized(ComputationOp):
    """
    Adds a fixed vector elementwise to a buffer.
    Compiles to an ADD_ELEMENTWISE instruction.
    """

    def __init__(self, shift_vector, in_place=False, name=None):
        super().__init__()
        self.shift_vector = np.asarray(shift_vector, dtype=np.float32)
        self.in_place = in_place
        self.name = name
        self._shift_tensor = tf.constant(self.shift_vector, dtype=tf.float32)
        self.keras_layer = layers.Lambda(
            lambda x: x + self._shift_tensor, name=name
        )

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("ShiftVectorized expects exactly one input.")
        output_tensor = self.keras_layer(inputs[0].tensor)
        return DataBuffer(output_tensor, op=self, inputs=inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        if len(input_indices) != 1:
            raise ValueError("ShiftVectorized expects exactly one input.")

        pw = weights_visited["parameters"]

        if self.in_place:
            target_index = input_indices[0]
        else:
            target_index = len(model_structure["buffer_sizes"])
            model_structure["buffer_sizes"].append(
                model_structure["buffer_sizes"][input_indices[0]]
            )
            copy_instr = {
                "type": "COPY",
                "input": input_indices[0],
                "output": target_index,
                "internal_index": 0,
            }
            model_structure["instructions"].append(copy_instr)

        if id(self) not in pw:
            pw[id(self)] = len(model_structure["parameters"])
            model_structure["parameters"].append(self.shift_vector.tolist())

        instr = {
            "type": "ADD_ELEMENTWISE",
            "input": target_index,
            "parameters": pw[id(self)],
        }
        model_structure["instructions"].append(instr)
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
            self.keras_layer = layers.Dense(
                self.a, name=self.name, activation="softmax"
            )

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

    def create_instruction_model(self, features=None, weights_visited=None):
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
