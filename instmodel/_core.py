"""Backend-agnostic graph ops and instruction emission shared by instmodel.tf and instmodel.torch.

Each backend subclasses these ops and supplies `_build(inputs)` (create the output DataBuffer),
`_linear_params()` (weight as (out, in), bias or None), `_to_numpy(t)` and, for normalization,
`_bn_stats()` -> (gamma, beta, moving_mean, moving_variance).
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import kendalltau

from .instruction_model import instruction_model_inference

NO_BATCH_NORM = 0
INPLACE = 1
NOT_INPLACE = 2

_ACTIVATION_CODES = {"r": "relu", "s": "sigmoid", "t": "tanh", "g": "gelu", "p": "softplus", "l": None}


def as_list(inputs):
    return inputs if isinstance(inputs, list) else [inputs]


def as_shape(shape_or_os):
    return (shape_or_os,) if isinstance(shape_or_os, int) else shape_or_os


def _new_buffer(ms, size):
    ms["buffer_sizes"].append(size)
    return len(ms["buffer_sizes"]) - 1


def _target(ms, src, in_place):
    if in_place:
        return src
    out = _new_buffer(ms, ms["buffer_sizes"][src])
    ms["instructions"].append({"type": "COPY", "input": src, "output": out, "internal_index": 0})
    return out


def _const(value):
    """Return (is_scalar, scalar, vector) for a constant given as a scalar or a vector."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
        return True, float(arr.flat[0]), None
    return False, None, arr


def _const_param(ms, is_scalar, scalar, vector, size):
    ms["parameters"].append([scalar] * size if is_scalar else vector.tolist())
    return len(ms["parameters"]) - 1


def _register_linear(op, weights_visited, ms):
    wv = weights_visited["weights"]
    if id(op) not in wv:
        wv[id(op)] = len(ms["weights"])
        weight, bias = op._linear_params()
        ms["weights"].append(weight.tolist())
        ms["bias"].append([0.0] * len(weight) if bias is None else bias.tolist())
    return wv[id(op)]


def _emit_map(op, weights_visited, ms, weight_matrix, ids, src, out, in_index, out_index, size):
    wm = op._to_numpy(weight_matrix).astype(np.float32)
    weights_visited["maps"][id(op)] = len(ms["maps"])
    ms["maps"].append({int(real_id): wm[bucket].tolist() for bucket, real_id in enumerate(ids)})
    ms["instructions"].append(
        {
            "type": "MAP_TRANSFORM",
            "input": src,
            "output": out,
            "internal_input_index": in_index,
            "internal_output_index": out_index,
            "map": len(ms["maps"]) - 1,
            "size": size,
            "default": wm[-1].tolist(),  # last bucket = unknown ids
        }
    )


def bucket_ids(train_ids, default_id):
    """Map each training id to its one-hot bucket; the extra last bucket is for unknown ids."""
    buckets = {}
    for _id in train_ids:
        i = int(_id)
        if i in buckets:
            raise ValueError(f"Duplicate ID found in train_ids: {_id}")
        buckets[i] = len(buckets)
    if default_id in buckets:
        raise ValueError(f"default_id ({default_id}) must not be in train_ids")
    return buckets


class DataBuffer:
    _copy_masked = None  # each backend sets its CopyMaskedComputation

    def __init__(self, op=None, inputs=None):
        self.op = op
        self.inputs = inputs if inputs is not None else []

    def __getitem__(self, key):
        if isinstance(key, int):
            indexes = [key]
        elif isinstance(key, slice):
            indexes = list(range(self.os))[key]
        elif isinstance(key, (list, tuple, np.ndarray)):
            indexes = list(key)
        else:
            raise TypeError("Unsupported index type for DataBuffer: " + str(type(key)))
        return self._copy_masked(indexes)([self])


class InputBuffer(DataBuffer):
    def __repr__(self):
        return f"InputBuffer(shape={self.shape})"

    def __call__(self, *args, **kwargs):
        return self


class ComputationOp(ABC):
    @abstractmethod
    def __call__(self, inputs):
        raise NotImplementedError("Subclasses must implement __call__.")

    @abstractmethod
    def compile_instructions(self, input_indices, weights_visited, model_structure) -> int:
        raise NotImplementedError("Subclasses must implement compile_instructions.")


class _Op(ComputationOp):
    def __call__(self, inputs):
        inputs = as_list(inputs)
        self._check(inputs)
        return self._build(inputs)

    def _check(self, inputs):
        pass

    def _out_size(self, inputs):
        return inputs[0].os


def _expect_one(items, message):
    if len(items) != 1:
        raise ValueError(message)


class ActivationComputation(_Op):
    def __init__(self, activation, in_place=False, name=None):
        super().__init__()
        self.in_place = in_place
        self.activation = activation.upper()
        self.name = name

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        _expect_one(input_indices, "ActivationComputation expects exactly one input.")
        target = _target(model_structure, input_indices[0], self.in_place)
        model_structure["instructions"].append(
            {"type": "ACTIVATION", "input": target, "activation": self.activation}
        )
        return target


class ReduceSum(_Op):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def _check(self, inputs):
        _expect_one(inputs, "ReduceSum expects exactly one input.")

    def _out_size(self, inputs):
        return 1

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        _expect_one(input_indices, "ReduceSum expects exactly one input.")
        out = _new_buffer(model_structure, 1)
        model_structure["instructions"].append(
            {"type": "REDUCE_SUM", "input": input_indices[0], "output": out}
        )
        return out


class Dense(_Op):
    def __init__(self, output_size, activation=None, use_bias=True, name=None):
        super().__init__()
        self.input_size = None
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        self.name = name

    def _check(self, inputs):
        _expect_one(inputs, "Dense expects exactly one input in the list.")
        if self.input_size is None:
            self.input_size = inputs[0].os

    def _out_size(self, inputs):
        return self.output_size

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        _expect_one(input_indices, "Dense.compile_instructions expects one input index.")
        weights = _register_linear(self, weights_visited, model_structure)
        out = _new_buffer(model_structure, self.output_size)
        instr = {"type": "DOT", "input": input_indices[0], "output": out, "weights": weights}
        if self.activation is not None:
            instr["activation"] = self.activation.upper()
        model_structure["instructions"].append(instr)
        return out


class CopyMaskedComputation(_Op):
    def __init__(self, indexes, name=None):
        super().__init__()
        self.indexes = indexes
        self.name = name

    def _out_size(self, inputs):
        return len(self.indexes)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        out = _new_buffer(model_structure, len(self.indexes))
        model_structure["instructions"].append(
            {"type": "COPY_MASKED", "input": input_indices[0], "output": out, "indexes": self.indexes}
        )
        return out


class Concatenate(_Op):
    def __init__(self, axis=-1, name=None):
        super().__init__()
        self.axis = axis
        self.name = name

    def _out_size(self, inputs):
        return sum(inp.os for inp in inputs)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        sizes = [model_structure["buffer_sizes"][i] for i in input_indices]
        out = _new_buffer(model_structure, sum(sizes))
        offset = 0
        for idx, size in zip(input_indices, sizes):
            model_structure["instructions"].append(
                {"type": "COPY", "input": idx, "output": out, "internal_index": offset}
            )
            offset += size
        return out


class MultiIdEmbeddings(_Op):
    """Replaces the id columns `feature_indexes` with learned embeddings appended after the other columns."""

    def __init__(self, feature_indexes, training_ids, output_dims, name=None):
        super().__init__()
        self.feature_indexes = feature_indexes
        self.training_ids = training_ids
        self.output_dims = output_dims
        self.name = name

    def _check(self, inputs):
        _expect_one(inputs, "MultiIdEmbeddings expects exactly one input buffer.")

    def _out_size(self, inputs):
        return inputs[0].os + self.layer.added_buffer_size

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        _expect_one(input_indices, "MultiIdEmbeddings.compile_instructions expects one input index.")
        src = input_indices[0]
        in_size = model_structure["buffer_sizes"][src]
        out = _new_buffer(model_structure, in_size + self.layer.added_buffer_size)
        keep = [i for i in range(in_size) if i not in self.feature_indexes]
        model_structure["instructions"].append(
            {"type": "COPY_MASKED", "input": src, "output": out, "indexes": keep}
        )
        out_index = len(keep)
        for k, encoder in enumerate(self.layer.singleIdEncoders):
            _emit_map(
                self, weights_visited, model_structure, encoder.weight_matrix, self.training_ids[k],
                src, out, self.feature_indexes[k], out_index, self.output_dims[k],
            )
            out_index += self.output_dims[k]
        return out


class SingleIdEmbeddings(_Op):
    """Learned embedding of the id at column `internal_input_index`; unknown ids share one vector."""

    def __init__(self, train_ids, output_dim, internal_input_index=0, name=None):
        super().__init__()
        self.output_dim = int(output_dim)
        self.default_id = -1
        self.train_ids = list(dict.fromkeys(train_ids))
        self.internal_input_index = internal_input_index
        self.name = name

    def _check(self, inputs):
        _expect_one(inputs, "SingleIdEmbeddings expects exactly one input buffer.")

    def _out_size(self, inputs):
        return self.output_dim

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        _expect_one(input_indices, "SingleIdEmbeddings.compile_instructions expects one input index.")
        out = _new_buffer(model_structure, self.output_dim)
        _emit_map(
            self, weights_visited, model_structure, self.layer.weight_matrix, self.train_ids,
            input_indices[0], out, self.internal_input_index, 0, self.output_dim,
        )
        return out


class NormalizationComputation(_Op):
    def __init__(self, in_place=False, center=True, scale=True, epsilon=1e-3, name=None):
        super().__init__()
        self.in_place = in_place
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.name = name

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        gamma, beta, moving_mean, moving_variance = self._bn_stats()
        std = gamma / np.sqrt(moving_variance + self.epsilon)
        target = _target(model_structure, input_indices[0], self.in_place)
        pw = weights_visited["parameters"]
        if id(self) not in pw:
            params = model_structure["parameters"]
            pw[id(self)] = [len(params), len(params) + 1, len(params) + 2]
            params += [(-moving_mean).tolist(), std.tolist(), beta.tolist()]
        for kind, p in zip(("ADD_ELEMENTWISE", "MUL_ELEMENTWISE", "ADD_ELEMENTWISE"), pw[id(self)]):
            model_structure["instructions"].append({"type": kind, "input": target, "parameters": p})
        return target


class _ConstOp(_Op):
    """Elementwise op with one constant operand (scalar broadcast or per-element vector)."""

    _label = _instr = None

    def _check(self, inputs):
        _expect_one(inputs, f"{self._label} expects exactly one input.")

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        _expect_one(input_indices, f"{self._label} expects exactly one input.")
        target = _target(model_structure, input_indices[0], self.in_place)
        pw = weights_visited["parameters"]
        if id(self) not in pw:
            size = model_structure["buffer_sizes"][input_indices[0]]
            pw[id(self)] = _const_param(
                model_structure, self._is_scalar, self._scalar_value, self._vector, size
            )
        model_structure["instructions"].append(
            {"type": self._instr, "input": target, "parameters": pw[id(self)]}
        )
        return target


class ScaleVectorized(_ConstOp):
    _label, _instr = "ScaleVectorized", "MUL_ELEMENTWISE"

    def __init__(self, scaling_vector, in_place=False, name=None):
        super().__init__()
        self._is_scalar, self._scalar_value, self.scaling_vector = _const(scaling_vector)
        self.in_place = in_place
        self.name = name

    @property
    def _vector(self):
        return self.scaling_vector


class ShiftVectorized(_ConstOp):
    _label, _instr = "ShiftVectorized", "ADD_ELEMENTWISE"

    def __init__(self, shift_vector, in_place=False, name=None):
        super().__init__()
        self._is_scalar, self._scalar_value, self.shift_vector = _const(shift_vector)
        self.in_place = in_place
        self.name = name

    @property
    def _vector(self):
        return self.shift_vector


class ClipVectorized(_Op):
    def __init__(self, clip_min=None, clip_max=None, in_place=False, name=None):
        super().__init__()
        if clip_min is None and clip_max is None:
            raise ValueError("ClipVectorized requires at least one of clip_min or clip_max.")
        self.in_place = in_place
        self.name = name
        self._has_min = clip_min is not None
        self._has_max = clip_max is not None
        none = (None, None, None)
        self._min_is_scalar, self._min_scalar, self.clip_min_vector = (
            _const(clip_min) if self._has_min else none
        )
        self._max_is_scalar, self._max_scalar, self.clip_max_vector = (
            _const(clip_max) if self._has_max else none
        )

    def _check(self, inputs):
        _expect_one(inputs, "ClipVectorized expects exactly one input.")

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        _expect_one(input_indices, "ClipVectorized expects exactly one input.")
        target = _target(model_structure, input_indices[0], self.in_place)
        pw = weights_visited["parameters"]
        if id(self) not in pw:
            size = model_structure["buffer_sizes"][input_indices[0]]
            cached = {}
            if self._has_min:
                cached["min"] = _const_param(
                    model_structure, self._min_is_scalar, self._min_scalar, self.clip_min_vector, size
                )
            if self._has_max:
                cached["max"] = _const_param(
                    model_structure, self._max_is_scalar, self._max_scalar, self.clip_max_vector, size
                )
            pw[id(self)] = cached
        instr = {"type": "CLIP_ELEMENTWISE", "input": target}
        for bound, p in pw[id(self)].items():
            instr[f"parameters_{bound}"] = p
        model_structure["instructions"].append(instr)
        return target


class Attention(_Op):
    """target * softmax(Dense(key)); inputs are [target, key]."""

    def __init__(self, name=None):
        super().__init__()
        self.a = None  # target (and output) size
        self.b = None  # key size
        self.name = name

    def _check(self, inputs):
        if len(inputs) != 2:
            raise ValueError("Attention expects two inputs: [target, key]")
        if self.a is None:
            self.a, self.b = inputs[0].os, inputs[1].os

    def _out_size(self, inputs):
        return self.a

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        weights = _register_linear(self, weights_visited, model_structure)
        out = _new_buffer(model_structure, self.a)
        model_structure["instructions"].append(
            {
                "type": "ATTENTION",
                "input": input_indices[0],
                "key": input_indices[1],
                "output": out,
                "weights": weights,
            }
        )
        return out


class _BufferReduce(_Op):
    """Elementwise reduction over equally sized input buffers."""

    _label = _verb = _instr = None

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        if not input_indices:
            raise ValueError(f"{self._label} expects at least one input.")
        sizes = model_structure["buffer_sizes"]
        if any(sizes[i] != sizes[input_indices[0]] for i in input_indices):
            raise ValueError(f"All inputs must have the same size for {self._verb}.")
        out = _new_buffer(model_structure, sizes[input_indices[0]])
        model_structure["instructions"].append(
            {"type": self._instr, "input": input_indices, "output": out}
        )
        return out


class Add(_BufferReduce):
    _label, _verb, _instr = "Add", "addition", "ADD_ELEMENTWISE_BUFFERS"

    def __init__(self, name=None):
        super().__init__()
        self.name = name


class Multiply(_BufferReduce):
    _label, _verb, _instr = "Multiply", "multiplication", "MULTIPLY_ELEMENTWISE_BUFFERS"

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def _check(self, inputs):
        if len(inputs) < 2:
            raise ValueError("Multiply expects at least two inputs.")


def _check_heads(data_size, heads_size):
    if data_size % heads_size != 0:
        raise ValueError(
            f"Data buffer size ({data_size}) must be divisible by heads buffer size ({heads_size})."
        )


class _Heads(_Op):
    """data (size H*D) combined with heads (size H): each head value is broadcast over its D elements."""

    _label = _instr = None

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.head_dim = None

    def _check(self, inputs):
        if len(inputs) != 2:
            raise ValueError(f"{self._label} expects exactly two inputs.")
        data_size, heads_size = inputs[0].os, inputs[1].os
        _check_heads(data_size, heads_size)
        self.head_dim = data_size // heads_size

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        if len(input_indices) != 2:
            raise ValueError(f"{self._label} expects exactly two inputs.")
        data_idx, heads_idx = input_indices
        data_size = model_structure["buffer_sizes"][data_idx]
        _check_heads(data_size, model_structure["buffer_sizes"][heads_idx])
        out = _new_buffer(model_structure, data_size)
        model_structure["instructions"].append(
            {"type": self._instr, "input": [data_idx, heads_idx], "output": out}
        )
        return out


class MultiplyHeads(_Heads):
    _label, _instr = "MultiplyHeads", "MULTIPLY_BUFFER_HEADS"


class AddHeads(_Heads):
    _label, _instr = "AddHeads", "ADD_BUFFER_HEADS"


class ModelGraph(ComputationOp):
    def __init__(self, input_buffers, output_buffer):
        super().__init__()
        self.input_buffers = as_list(input_buffers)
        self.output_buffer = output_buffer

    @property
    def os(self):
        return self.output_buffer.os

    def _emit(self, visited, weights_visited, ms, nested):
        def visit(buffer):
            idx = visited.get(id(buffer))
            if idx is not None:
                return idx
            if buffer.op is not None:
                input_idx = [visit(inp) for inp in buffer.inputs]
                idx = buffer.op.compile_instructions(input_idx, weights_visited, ms)
            elif nested and isinstance(buffer, InputBuffer):
                raise ValueError(f"{buffer} is not an input of the nested ModelGraph.")
            else:
                idx = _new_buffer(ms, buffer.os)
            visited[id(buffer)] = idx
            return idx

        return visit(self.output_buffer)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        visited = {}
        for buffer, idx in zip(self.input_buffers, input_indices):
            visited.setdefault(id(buffer), idx)
        return self._emit(visited, weights_visited, model_structure, nested=True)

    def create_instruction_model(self, features=None, weights_visited=None):
        ms = {
            "features": features or [],
            "buffer_sizes": [],
            "instructions": [],
            "maps": [],
            "weights": [],
            "bias": [],
            "parameters": [],
        }
        if weights_visited is None:
            weights_visited = {"weights": {}, "parameters": {}, "maps": {}}
        visited = {}
        for buffer in self.input_buffers:
            if id(buffer) not in visited:
                visited[id(buffer)] = _new_buffer(ms, buffer.os)
        self._emit(visited, weights_visited, ms, nested=False)
        return ms


def ff_model(ns, sizes, use_batch_norm=NO_BATCH_NORM, activations=None):
    features_len, *hidden_sizes, last_layer_size = sizes
    if activations is None:
        activations = ["relu"] * len(hidden_sizes) + ["sigmoid"]
    elif len(activations) != len(hidden_sizes) + 1:
        raise ValueError("The number of activations must match the number of hidden layers + 1.")
    if isinstance(activations, str):
        activations = [_ACTIVATION_CODES[a] for a in activations]
    inp = x = ns.InputBuffer(features_len)
    if use_batch_norm != NO_BATCH_NORM:
        x = ns.NormalizationComputation(in_place=use_batch_norm == INPLACE)(x)
    for size, activation in zip([*hidden_sizes, last_layer_size], activations):
        x = ns.Dense(size, activation=activation)(x)
    return ns.ModelGraph(inp, x)


def create_stair_structure(ns, features_len, hidden_sizes=None, use_batch_norm=False):
    """Dense stack whose output concatenates the (optionally normalized) input and every hidden layer."""
    inp = ns.InputBuffer(features_len)
    ops, x = [], inp
    if use_batch_norm:
        ops.append(ns.NormalizationComputation())
        x = ops[-1](x)
    buffers = [x]
    for size in hidden_sizes or [14, 12, 10]:
        ops.append(ns.Dense(size, activation="relu"))
        x = ops[-1](x)
        buffers.append(x)
    out = ns.Concatenate()(buffers) if len(buffers) > 1 else x
    return ns.ModelGraph(inp, out), ops


def generate_validation_data(features, model, means=None, stds=None):
    """Random inputs (optionally scaled by stds and shifted by means) with the model's outputs."""
    input_data = np.random.randn(10, len(features)).astype(np.float32)
    if stds is not None:
        input_data = input_data * (np.array(stds) + 1e-6)
    if means is not None:
        input_data = input_data + np.array(means)
    output_data = model.predict_on_batch(input_data)
    return {"inputs": input_data.tolist(), "expected_outputs": output_data.tolist()}


def tau_compare(predictions, y_data):
    """Kendall tau-b per column (NaN -> 0.0); a float when there is a single column."""
    results = []
    for col in range(y_data.shape[1]):
        tau, _ = kendalltau(predictions[:, col], y_data[:, col])
        results.append(0.0 if np.isnan(tau) else tau)
    return results if len(results) > 1 else results[0]


def _as_2d(a):
    if hasattr(a, "to_numpy"):
        a = a.to_numpy()
    return a.reshape(-1, 1) if a.ndim == 1 else a


def score_selection(model, x_data, y_data):
    """tau_compare of a model (anything with .predict, or an instruction model dict) against y_data."""
    y_data = _as_2d(y_data)
    if hasattr(model, "predict"):
        predictions = model.predict(x_data)
    else:
        predictions = instruction_model_inference(model, x_data)[-1]
    return tau_compare(_as_2d(predictions), y_data)


def validate_predictions(model, validation_data, label):
    x_val = np.array(validation_data["inputs"])
    y_expected = np.array(validation_data["expected_outputs"])
    y_pred = model.predict(x_val)
    if np.allclose(y_expected, y_pred, atol=1e-6):
        print(f"{label} model validation successful: predictions match expected outputs.")
        return
    print(f"{label} model validation failed.")
    print("Expected outputs:", y_expected)
    print("Predictions:", y_pred)
    raise AssertionError(f"{label} model validation failed.")
