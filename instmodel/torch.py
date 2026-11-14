import math
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

_ACTIVATIONS = {
    "RELU": F.relu,
    "SIGMOID": torch.sigmoid,
    "TANH": torch.tanh,
    "SOFTMAX": lambda x: F.softmax(x, dim=-1),
    "GELU": F.gelu,
    "SOFTPLUS": F.softplus,
    "SQRT": lambda x: torch.where(x > 0, torch.sqrt(x), torch.zeros_like(x)),
    "LOG": lambda x: torch.where(x > 0, torch.log(x + 1), torch.zeros_like(x)),
    "LOG10": lambda x: torch.where(x > 0, torch.log(x + 1) / math.log(10.0), torch.zeros_like(x)),
    "INVERSE": lambda x: 1 - x,
    "EXP": lambda x: torch.exp(torch.clamp(x, -88, 88)),
    "SIGN": torch.sign,
}


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    fn = _ACTIVATIONS.get(activation.upper())
    if fn is None:
        raise ValueError(f"Unexpected activation: {activation}")
    return fn(x)


def create_stair_structure(
    features_len: int,
    hidden_sizes: Optional[list[int]] = None,
    use_batch_norm: bool = False,
):
    """Returns (ModelGraph, the normalization/Dense ops)."""
    return _core.create_stair_structure(_this, features_len, hidden_sizes, use_batch_norm)


def ff_model(sizes: list[int], use_batch_norm: int = 0, activations=None):
    """sizes = [features, *hidden, out]; activations: list or codes string (r,s,t,g,p,l=linear)."""
    return _core.ff_model(_this, sizes, use_batch_norm, activations)


class OneHotDenseEncoderTorch(nn.Module):
    """Embeds float-encoded integer ids: round -> one-hot bucket (last = unknown) -> bias-less Linear."""

    def __init__(self, train_ids: List[int], output_dim: int, default_id: int = -1):
        super().__init__()
        self.default_id = int(default_id)
        self._id_to_bucket = _core.bucket_ids(train_ids, self.default_id)
        self.oov_bucket = len(self._id_to_bucket)
        self.depth = self.oov_bucket + 1
        self.dense = nn.Linear(self.depth, output_dim, bias=False)

    def forward(self, raw_ids: torch.Tensor) -> torch.Tensor:
        int_ids = torch.round(raw_ids).long().reshape(-1)
        bucket_idx = torch.full_like(int_ids, self.oov_bucket)
        for id_, bucket in self._id_to_bucket.items():
            bucket_idx = torch.where(
                int_ids == id_,
                torch.tensor(bucket, dtype=torch.long, device=int_ids.device),
                bucket_idx,
            )
        return self.dense(F.one_hot(bucket_idx, num_classes=self.depth).float())

    @property
    def weight_matrix(self) -> torch.Tensor:
        """(depth, output_dim) weight."""
        return self.dense.weight.T


class MultiOneHotDenseEncoderTorch(nn.Module):
    """Keeps non-id columns and appends one OneHotDenseEncoderTorch embedding per id column."""

    def __init__(self, feature_indexes: List[int], training_ids: List[List[int]], output_dims: List[int]):
        super().__init__()
        self.feature_indexes = feature_indexes
        self.training_ids = training_ids
        self.output_dims = output_dims
        self.singleIdEncoders = nn.ModuleList(
            [OneHotDenseEncoderTorch(train_ids=ids, output_dim=dim) for ids, dim in zip(training_ids, output_dims)]
        )
        self.added_buffer_size = sum(output_dims) - len(feature_indexes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        keep = [i for i in range(int(input_tensor.shape[1])) if i not in self.feature_indexes]
        encoded = [enc(input_tensor[:, i]) for i, enc in zip(self.feature_indexes, self.singleIdEncoders)]
        return torch.cat([input_tensor[:, keep]] + encoded, dim=-1)


class DataBuffer(_core.DataBuffer):
    """Output shape plus the op and input buffers that produced it."""

    def __init__(self, shape, op=None, inputs=None):
        super().__init__(op, inputs)
        self._shape = shape

    def __repr__(self):
        return f"DataBuffer(shape={self._shape})"

    @property
    def os(self):
        return self._shape[-1]


class InputBuffer(_core.InputBuffer, DataBuffer):
    def __init__(self, shape_or_os, name=None):
        self.shape = _core.as_shape(shape_or_os)
        super().__init__(self.shape)


class ComputationOp(_core.ComputationOp):
    """Base op; subclasses implement __call__, compile_instructions and forward_op(*tensors)."""

    def __init__(self):
        self.torch_module = None

    def forward_op(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward_op.")

    def _build(self, inputs):
        return DataBuffer((self._out_size(inputs),), op=self, inputs=inputs)

    def _module(self):
        """nn.Module holding this op's trainable state, if any."""
        return None

    def _linear_params(self):
        m = self.torch_module
        return self._to_numpy(m.weight), (None if m.bias is None else self._to_numpy(m.bias))

    @staticmethod
    def _to_numpy(t):
        return t.detach().cpu().numpy()


def _const(x, is_scalar, scalar, vector):
    return scalar if is_scalar else torch.tensor(vector, dtype=x.dtype, device=x.device)


class ActivationComputation(_core.ActivationComputation, ComputationOp):
    def forward_op(self, x):
        return _apply_activation(x, self.activation)


class ReduceSum(_core.ReduceSum, ComputationOp):
    def forward_op(self, x):
        return torch.sum(x, dim=-1, keepdim=True)


class Dense(_core.Dense, ComputationOp):
    def _build(self, inputs):
        if self.torch_module is None:
            self.torch_module = nn.Linear(self.input_size, self.output_size, bias=self.use_bias)
        return super()._build(inputs)

    def _module(self):
        return self.torch_module

    def forward_op(self, x):
        out = self.torch_module(x)
        return out if self.activation is None else _apply_activation(out, self.activation)


class CopyMaskedComputation(_core.CopyMaskedComputation, ComputationOp):
    def forward_op(self, x):
        return x[:, self.indexes]


DataBuffer._copy_masked = CopyMaskedComputation


class Concatenate(_core.Concatenate, ComputationOp):
    def forward_op(self, *input_tensors):
        return torch.cat(input_tensors, dim=-1)


class MultiIdEmbeddings(_core.MultiIdEmbeddings, ComputationOp):
    def __init__(
        self,
        feature_indexes: List[int],
        training_ids: List[List[int]],
        output_dims: List[int],
        name: Optional[str] = None,
    ):
        super().__init__(feature_indexes, training_ids, output_dims, name)
        self.layer = MultiOneHotDenseEncoderTorch(
            feature_indexes=feature_indexes, training_ids=training_ids, output_dims=output_dims
        )

    def _module(self):
        return self.layer

    def forward_op(self, x):
        return self.layer(x)


class SingleIdEmbeddings(_core.SingleIdEmbeddings, ComputationOp):
    def __init__(
        self,
        train_ids: List[int],
        output_dim: int,
        internal_input_index: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(train_ids, output_dim, internal_input_index, name)
        self.layer = OneHotDenseEncoderTorch(
            train_ids=train_ids, output_dim=output_dim, default_id=self.default_id
        )

    def _module(self):
        return self.layer

    def forward_op(self, x):
        return self.layer(x)


class NormalizationComputation(_core.NormalizationComputation, ComputationOp):
    def __init__(self, in_place=False, center=True, scale=True, epsilon=1e-3, name=None):
        super().__init__(in_place, center, scale, epsilon, name)
        self._bn = None

    def _ensure_bn(self, num_features):
        if self._bn is None:
            self._bn = nn.BatchNorm1d(num_features, eps=self.epsilon, affine=(self.center or self.scale))
            if self._bn.weight is not None and not self.scale:
                self._bn.weight.requires_grad_(False)
                self._bn.weight.fill_(1.0)
            if self._bn.bias is not None and not self.center:
                self._bn.bias.requires_grad_(False)
                self._bn.bias.fill_(0.0)

    def _build(self, inputs):
        self._ensure_bn(inputs[0].os)
        return super()._build(inputs)

    def _module(self):
        return self._bn

    def forward_op(self, x):
        return self._bn(x)

    def _bn_stats(self):
        bn = self._bn
        mean = self._to_numpy(bn.running_mean)
        var = self._to_numpy(bn.running_var)
        gamma = self._to_numpy(bn.weight) if bn.weight is not None and self.scale else np.ones_like(mean)
        beta = self._to_numpy(bn.bias) if bn.bias is not None and self.center else np.zeros_like(gamma)
        return gamma, beta, mean, var


class ScaleVectorized(_core.ScaleVectorized, ComputationOp):
    def forward_op(self, x):
        return x * _const(x, self._is_scalar, self._scalar_value, self.scaling_vector)


class ShiftVectorized(_core.ShiftVectorized, ComputationOp):
    def forward_op(self, x):
        return x + _const(x, self._is_scalar, self._scalar_value, self.shift_vector)


class ClipVectorized(_core.ClipVectorized, ComputationOp):
    def forward_op(self, x):
        if self._has_min:
            if self._min_is_scalar:
                x = torch.clamp(x, min=self._min_scalar)
            else:
                x = torch.maximum(x, _const(x, False, None, self.clip_min_vector))
        if self._has_max:
            if self._max_is_scalar:
                x = torch.clamp(x, max=self._max_scalar)
            else:
                x = torch.minimum(x, _const(x, False, None, self.clip_max_vector))
        return x


class Attention(_core.Attention, ComputationOp):
    def _build(self, inputs):
        if self.torch_module is None:
            self.torch_module = nn.Linear(self.b, self.a)
        return super()._build(inputs)

    def _module(self):
        return self.torch_module

    def forward_op(self, target, key):
        return target * F.softmax(self.torch_module(key), dim=-1)


class Add(_core.Add, ComputationOp):
    def forward_op(self, *input_tensors):
        return torch.sum(torch.stack(input_tensors, dim=0), dim=0)


class Multiply(_core.Multiply, ComputationOp):
    def forward_op(self, *input_tensors):
        return torch.prod(torch.stack(input_tensors, dim=0), dim=0)


class MultiplyHeads(_core.MultiplyHeads, ComputationOp):
    def forward_op(self, data, heads):
        return data * torch.repeat_interleave(heads, self.head_dim, dim=-1)


class AddHeads(_core.AddHeads, ComputationOp):
    def forward_op(self, data, heads):
        return data + torch.repeat_interleave(heads, self.head_dim, dim=-1)


class _InternalModule(nn.Module):
    """Runs the op graph in topological order; registers each op's trainable module once."""

    def __init__(self, input_buffers, output_buffer):
        super().__init__()
        self._input_buffers = _core.as_list(input_buffers)
        self._output_buffer = output_buffer
        self._topo_order = []
        self._modules_dict = nn.ModuleDict()
        self._build_topo(output_buffer, set())

    def _build_topo(self, buffer, visited):
        if id(buffer) in visited:
            return
        visited.add(id(buffer))
        if isinstance(buffer, InputBuffer):
            return
        for inp in buffer.inputs:
            self._build_topo(inp, visited)
        if buffer.op is not None:
            self._topo_order.append(buffer)
            self._register_op_modules(buffer.op)

    def _register_op_modules(self, op):
        op_id = str(id(op))
        if op_id not in self._modules_dict and isinstance(op, ComputationOp):
            module = op._module()
            if module is not None:
                self._modules_dict[op_id] = module

    def forward(self, *inputs):
        tensor_map = {id(buf): x for buf, x in zip(self._input_buffers, inputs)}
        for buffer in self._topo_order:
            tensor_map[id(buffer)] = buffer.op.forward_op(*[tensor_map[id(b)] for b in buffer.inputs])
        return tensor_map[id(self._output_buffer)]


_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}
_LOSSES = {
    "mse": nn.MSELoss,
    "binary_crossentropy": nn.BCELoss,
    "bce": nn.BCELoss,
    "crossentropy": nn.CrossEntropyLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "mae": nn.L1Loss,
    "l1": nn.L1Loss,
}


class ModelGraph(_core.ModelGraph, ComputationOp):
    """Graph from input buffers to an output buffer with a Keras-like compile/fit/predict API."""

    def __init__(self, input_buffers, output_buffer: DataBuffer, name=None, device=None):
        super().__init__(input_buffers, output_buffer)
        self._device = device or _default_device()
        self._torch_module = _InternalModule(self.input_buffers, output_buffer).to(self._device)
        self._optimizer = None
        self._loss_fn = None
        self._compiled_module = None

    def get_torch(self):
        return self._torch_module

    def get_module(self):
        return self._torch_module

    def _module(self):
        return self._torch_module

    def forward_op(self, *input_tensors):
        return self._torch_module(*input_tensors)

    def compile(self, optimizer="adam", loss="mse", lr=0.001, metrics=None, **kwargs):
        self._metrics = metrics or []
        if isinstance(optimizer, str):
            opt_cls = _OPTIMIZERS.get(optimizer.lower())
            if opt_cls is None:
                raise ValueError(f"Unknown optimizer: {optimizer}")
            optimizer = opt_cls(self._torch_module.parameters(), lr=lr, **kwargs)
        self._optimizer = optimizer
        if isinstance(loss, str):
            loss_cls = _LOSSES.get(loss.lower())
            if loss_cls is None:
                raise ValueError(f"Unknown loss: {loss}")
            loss = loss_cls()
        self._loss_fn = loss
        self._compiled_module = torch.compile(self._torch_module)
        return self._compiled_module

    def _active_module(self):
        return self._torch_module if self._compiled_module is None else self._compiled_module

    def _tensor(self, a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=self._device)

    def _tensors(self, x):
        return [self._tensor(xi) for xi in x] if isinstance(x, list) else [self._tensor(x)]

    def fit(self, x, y, epochs=1, batch_size=32, verbose=1, shuffle=True, sample_weight=None):
        """Returns the per-epoch mean loss; sample_weight switches the loss to weighted MSE."""
        module = self._active_module()
        module.train()
        xs = self._tensors(x)
        y_all = self._tensor(y)
        w_all = None if sample_weight is None else self._tensor(sample_weight)
        n_samples = xs[0].shape[0]
        history = []
        for epoch in range(epochs):
            xe, ye, we = xs, y_all, w_all
            if shuffle:
                perm = torch.randperm(n_samples, device=self._device)
                xe, ye = [t[perm] for t in xs], y_all[perm]
                we = None if w_all is None else w_all[perm]
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                y_batch = ye[start:end]
                self._optimizer.zero_grad()
                pred = module(*[t[start:end] for t in xe])
                if we is None:
                    loss = self._loss_fn(pred, y_batch)
                else:
                    loss = ((pred - y_batch).pow(2).mean(dim=-1) * we[start:end]).mean()
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / n_batches
            history.append(avg_loss)
            if verbose:
                fmt = ".4f" if abs(avg_loss) > 1e-3 else ".4e"
                print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:{fmt}}")
        return history

    def predict(self, x, verbose=0):
        module = self._active_module()
        module.eval()
        with torch.no_grad():
            pred = module(*self._tensors(x))
        return pred.cpu().numpy()

    def predict_on_batch(self, x):
        return self.predict(x)

    def summary(self):
        print(self._torch_module)

    def __call__(self, inputs):
        return DataBuffer(self.output_buffer._shape, op=self, inputs=_core.as_list(inputs))


def create_model_graph(inputs, output: DataBuffer) -> ModelGraph:
    return ModelGraph(_core.as_list(inputs), output)


def create_instruction_model(inputs, output: DataBuffer):
    return create_model_graph(inputs, output).create_instruction_model()


def validate_torch_model(model, validation_data):
    """Raises AssertionError unless model.predict matches validation_data (atol 1e-6)."""
    _core.validate_predictions(model, validation_data, "PyTorch")


validate_model = validate_torch_model
