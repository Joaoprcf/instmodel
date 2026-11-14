import sys
from typing import List, Optional

import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model

from . import _core
from ._core import (  # noqa: F401  (re-exported public API)
    INPLACE,
    NO_BATCH_NORM,
    NOT_INPLACE,
    generate_validation_data,
    score_selection,
    tau_compare,
)
from .instruction_model import instruction_model_inference  # noqa: F401

_this = sys.modules[__name__]

_NATIVE_ACTIVATIONS = {"RELU", "SIGMOID", "TANH", "SOFTMAX", "GELU", "SOFTPLUS"}
_LAMBDA_ACTIVATIONS = {
    "SQRT": lambda x: tf.where(x > 0, tf.sqrt(x), tf.zeros_like(x)),
    "LOG": lambda x: tf.math.log(tf.maximum(x, 0) + 1),
    "LOG10": lambda x: tf.math.log(tf.maximum(x, 0) + 1) / tf.math.log(10.0),
    "INVERSE": lambda x: 1 - x,
    "EXP": lambda x: tf.exp(tf.clip_by_value(x, -88, 88)),
    "SIGN": lambda x: tf.sign(x),
}


def create_stair_structure(
    features_len: int,
    hidden_sizes: Optional[list[int]] = None,
    use_batch_norm: bool = False,
):
    """Returns (ModelGraph, keras layers of the normalization/Dense ops)."""
    model, ops = _core.create_stair_structure(_this, features_len, hidden_sizes, use_batch_norm)
    return model, [op.keras_layer for op in ops]


def ff_model(sizes: list[int], use_batch_norm: int = 0, activations=None):
    """sizes = [features, *hidden, out]; activations: list or codes string (r,s,t,g,p,l=linear)."""
    return _core.ff_model(_this, sizes, use_batch_norm, activations)


class OneHotDenseEncoder(tf.keras.layers.Layer):
    """Embeds float-encoded integer ids: round -> one-hot bucket (last = unknown) -> bias-less Dense."""

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
        self._id_to_bucket = _core.bucket_ids(train_ids, self.default_id)
        self.oov_bucket = len(self._id_to_bucket)
        self.depth = self.oov_bucket + 1
        self.dense = layers.Dense(output_dim, use_bias=False, dtype=tf.float32)

    def call(self, raw_ids):
        int_ids = tf.reshape(tf.cast(tf.math.round(raw_ids), self.key_dtype), [-1])
        bucket_idx = tf.fill(tf.shape(int_ids), tf.constant(self.oov_bucket, dtype=self.key_dtype))
        for id_, bucket in self._id_to_bucket.items():
            bucket_idx = tf.where(
                tf.equal(int_ids, tf.constant(id_, dtype=self.key_dtype)),
                tf.constant(bucket, dtype=self.key_dtype),
                bucket_idx,
            )
        return self.dense(tf.one_hot(bucket_idx, depth=self.depth, dtype=tf.float32))

    @property
    def os(self):
        return self.dense.units

    @property
    def weight_matrix(self) -> tf.Tensor:
        """(depth, output_dim) kernel."""
        return self.dense.kernel


class MultiOneHotDenseEncoder(tf.keras.layers.Layer):
    """Keeps non-id columns and appends one OneHotDenseEncoder embedding per id column."""

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
            OneHotDenseEncoder(train_ids=ids, output_dim=dim)
            for ids, dim in zip(training_ids, output_dims)
        ]
        self.added_buffer_size = sum(output_dims) - len(feature_indexes)

    def call(self, inputs):
        inputs = _core.as_list(inputs)
        if len(inputs) != 1:
            raise ValueError("Dense expects exactly one input in the list.")
        x = inputs[0]
        keep = [i for i in range(int(x.shape[1])) if i not in self.feature_indexes]
        encoded = [enc(x[:, i]) for i, enc in zip(self.feature_indexes, self.singleIdEncoders)]
        return tf.concat([tf.gather(x, keep, axis=1)] + encoded, axis=-1)


class DataBuffer(_core.DataBuffer):
    """Symbolic Keras tensor plus the op and input buffers that produced it."""

    def __init__(self, tensor, op=None, inputs=None):
        super().__init__(op, inputs)
        self.tensor = tensor

    def __repr__(self):
        return f"DataBuffer(shape={self.tensor.shape})"

    @property
    def os(self):
        return int(self.tensor.shape[-1])


class InputBuffer(_core.InputBuffer, DataBuffer):
    def __init__(self, shape_or_os, name=None):
        self.shape = _core.as_shape(shape_or_os)
        super().__init__(layers.Input(shape=self.shape, name=name))


class ComputationOp(_core.ComputationOp):
    """Base op; subclasses implement __call__(list[DataBuffer]) and compile_instructions."""

    _multi_input = False

    def __init__(self):
        self.keras_layer = None

    def _build(self, inputs):
        x = [inp.tensor for inp in inputs] if self._multi_input else inputs[0].tensor
        return DataBuffer(self.keras_layer(x), op=self, inputs=inputs)

    def _linear_params(self):
        w = self.keras_layer.get_weights()
        return w[0].T, (w[1] if len(w) > 1 else None)

    @staticmethod
    def _to_numpy(t):
        return t.numpy()


class ActivationComputation(_core.ActivationComputation, ComputationOp):
    def __init__(self, activation, in_place=False, name: Optional[str] = None):
        super().__init__(activation, in_place, name)
        if self.activation in _NATIVE_ACTIVATIONS:
            self.keras_layer = tf.keras.layers.Activation(self.activation.lower(), name=name)
        elif self.activation in _LAMBDA_ACTIVATIONS:
            self.keras_layer = layers.Lambda(_LAMBDA_ACTIVATIONS[self.activation], name=name)
        else:
            raise ValueError(f"Unexpected activation: {self.activation}")


class ReduceSum(_core.ReduceSum, ComputationOp):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.keras_layer = layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name=name
        )


class Dense(_core.Dense, ComputationOp):
    def __init__(self, output_size, activation=None, use_bias=True, name=None):
        super().__init__(output_size, activation, use_bias, name)
        self.keras_layer = layers.Dense(
            output_size, activation=activation, use_bias=use_bias, name=name
        )


class CopyMaskedComputation(_core.CopyMaskedComputation, ComputationOp):
    def _build(self, inputs):
        idx = tf.constant(self.indexes, dtype=tf.int32)
        gathered = layers.Lambda(lambda x: tf.gather(x, indices=idx, axis=1))(inputs[0].tensor)
        return DataBuffer(gathered, op=self, inputs=inputs)


DataBuffer._copy_masked = CopyMaskedComputation


class Concatenate(_core.Concatenate, ComputationOp):
    _multi_input = True

    def __init__(self, axis=-1, name=None):
        super().__init__(axis, name)
        self.keras_layer = layers.Concatenate(axis=axis, name=name)


class MultiIdEmbeddings(_core.MultiIdEmbeddings, ComputationOp):
    def __init__(
        self,
        feature_indexes: List[int],
        training_ids: List[List[int]],
        output_dims: List[int],
        name: Optional[str] = None,
    ):
        super().__init__(feature_indexes, training_ids, output_dims, name)
        self.layer = self.keras_layer = MultiOneHotDenseEncoder(
            feature_indexes=feature_indexes, training_ids=training_ids, output_dims=output_dims
        )


class SingleIdEmbeddings(_core.SingleIdEmbeddings, ComputationOp):
    def __init__(
        self,
        train_ids: List[int],
        output_dim: int,
        internal_input_index: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(train_ids, output_dim, internal_input_index, name)
        self.layer = self.keras_layer = OneHotDenseEncoder(
            train_ids=train_ids, output_dim=output_dim, default_id=self.default_id
        )


class NormalizationComputation(_core.NormalizationComputation, ComputationOp):
    def __init__(self, in_place=False, center=True, scale=True, epsilon=1e-3, name=None):
        super().__init__(in_place, center, scale, epsilon, name)
        self.keras_layer = tf.keras.layers.BatchNormalization(
            epsilon=epsilon, center=center, scale=scale, axis=-1, name=name
        )

    def _bn_stats(self):
        weights = self.keras_layer.get_weights()
        if len(weights) == 4:
            return weights
        if len(weights) == 3:
            if self.keras_layer.center and not self.keras_layer.scale:
                beta, mean, var = weights
                return np.ones_like(mean), beta, mean, var
            gamma, mean, var = weights
            return gamma, np.zeros_like(gamma), mean, var
        if len(weights) == 2:
            mean, var = weights
            return np.ones_like(mean), np.zeros_like(mean), mean, var
        raise ValueError(
            f"Unexpected number of BN weights returned: {len(weights)}. "
            "Check your 'center' and 'scale' arguments."
        )


def _const_tensor(is_scalar, scalar, vector):
    return tf.constant(scalar if is_scalar else vector, dtype=tf.float32)


class ScaleVectorized(_core.ScaleVectorized, ComputationOp):
    def __init__(self, scaling_vector, in_place=False, name=None):
        super().__init__(scaling_vector, in_place, name)
        self._scaling_tensor = _const_tensor(self._is_scalar, self._scalar_value, self.scaling_vector)
        self.keras_layer = layers.Lambda(lambda x: x * self._scaling_tensor, name=name)


class ShiftVectorized(_core.ShiftVectorized, ComputationOp):
    def __init__(self, shift_vector, in_place=False, name=None):
        super().__init__(shift_vector, in_place, name)
        self._shift_tensor = _const_tensor(self._is_scalar, self._scalar_value, self.shift_vector)
        self.keras_layer = layers.Lambda(lambda x: x + self._shift_tensor, name=name)


class ClipVectorized(_core.ClipVectorized, ComputationOp):
    def __init__(self, clip_min=None, clip_max=None, in_place=False, name=None):
        super().__init__(clip_min, clip_max, in_place, name)
        min_t = self._min_tensor = (
            _const_tensor(self._min_is_scalar, self._min_scalar, self.clip_min_vector)
            if self._has_min
            else None
        )
        max_t = self._max_tensor = (
            _const_tensor(self._max_is_scalar, self._max_scalar, self.clip_max_vector)
            if self._has_max
            else None
        )

        def clip_fn(x):
            if min_t is not None:
                x = tf.maximum(x, min_t)
            if max_t is not None:
                x = tf.minimum(x, max_t)
            return x

        self.keras_layer = layers.Lambda(clip_fn, name=name)


class Attention(_core.Attention, ComputationOp):
    def _build(self, inputs):
        target, key = inputs
        if self.keras_layer is None:
            self.keras_layer = layers.Dense(self.a, name=self.name, activation="softmax")
        return DataBuffer(target.tensor * self.keras_layer(key.tensor), op=self, inputs=inputs)


class Add(_core.Add, ComputationOp):
    _multi_input = True

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.keras_layer = tf.keras.layers.Add(name=name)


class Multiply(_core.Multiply, ComputationOp):
    _multi_input = True

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.keras_layer = tf.keras.layers.Multiply(name=name)


class _TfHeads(ComputationOp):
    _multi_input = True
    _combine = None

    def _build(self, inputs):
        head_dim, combine = self.head_dim, self._combine
        self.keras_layer = layers.Lambda(
            lambda x: combine(x[0], tf.repeat(x[1], repeats=head_dim, axis=-1)), name=self.name
        )
        return super()._build(inputs)


class MultiplyHeads(_core.MultiplyHeads, _TfHeads):
    _combine = staticmethod(tf.multiply)


class AddHeads(_core.AddHeads, _TfHeads):
    _combine = staticmethod(tf.add)


class ModelGraph(_core.ModelGraph, ComputationOp):
    """Graph from input buffers to an output buffer; trains as a Keras Model and exports instructions."""

    def __init__(self, input_buffers, output_buffer: DataBuffer, name=None):
        super().__init__(input_buffers, output_buffer)
        self._keras_model = Model(
            inputs=[buf.tensor for buf in self.input_buffers],
            outputs=self.output_buffer.tensor,
            name=name,
        )

    def get_keras(self):
        return self._keras_model

    def get_module(self):
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
        inputs = _core.as_list(inputs)
        return DataBuffer(self._keras_model([inp.tensor for inp in inputs]), op=self, inputs=inputs)


def create_model_graph(inputs, output: DataBuffer) -> ModelGraph:
    return ModelGraph(_core.as_list(inputs), output)


def create_instruction_model(inputs, output: DataBuffer):
    return create_model_graph(inputs, output).create_instruction_model()


def validate_keras_model(keras_model, validation_data):
    """Raises AssertionError unless keras_model.predict matches validation_data (atol 1e-6)."""
    _core.validate_predictions(keras_model, validation_data, "Keras")


validate_model = validate_keras_model
