#!/usr/bin/env python3
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .instruction_model import (
    instruction_model_inference,
)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    act = activation.upper()
    if act == "RELU":
        return F.relu(x)
    elif act == "SIGMOID":
        return torch.sigmoid(x)
    elif act == "TANH":
        return torch.tanh(x)
    elif act == "SOFTMAX":
        return F.softmax(x, dim=-1)
    elif act == "GELU":
        return F.gelu(x)
    elif act == "SOFTPLUS":
        return F.softplus(x)
    elif act == "SQRT":
        return torch.where(x > 0, torch.sqrt(x), torch.zeros_like(x))
    elif act == "LOG":
        return torch.where(x > 0, torch.log(x + 1), torch.zeros_like(x))
    elif act == "LOG10":
        return torch.where(
            x > 0,
            torch.log(x + 1) / math.log(10.0),
            torch.zeros_like(x),
        )
    elif act == "INVERSE":
        return 1 - x
    else:
        raise ValueError(f"Unexpected activation: {activation}")


def create_stair_structure(
    features_len: int,
    hidden_sizes: Optional[list[int]] = None,
    use_batch_norm: bool = False,
):
    hidden_sizes = hidden_sizes or [14, 12, 10]
    input_layer = InputBuffer(features_len)
    normalizer_layer = NormalizationComputation()
    comp_layers, current_buffer = (
        ([normalizer_layer], normalizer_layer(input_layer))
        if use_batch_norm
        else ([], input_layer)
    )

    buffers = [current_buffer]

    for size in hidden_sizes:
        mid_layer = Dense(size, activation="relu")
        mid_buffer = mid_layer(current_buffer)
        comp_layers.append(mid_layer)
        buffers.append(mid_buffer)
        current_buffer = mid_buffer

    current_buffer = Concatenate()(buffers) if len(buffers) > 1 else current_buffer
    model = ModelGraph(input_layer, current_buffer)

    return model, comp_layers


NO_BATCH_NORM = 0
INPLACE = 1
NOT_INPLACE = 2


def ff_model(sizes: list[int], use_batch_norm: int = 0, activations=None):
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
            "g": "gelu",
            "p": "softplus",
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


###############################################################################
# Encoder layers (nn.Module)
###############################################################################

class OneHotDenseEncoderTorch(nn.Module):
    def __init__(
        self,
        train_ids: List[int],
        output_dim: int,
        default_id: int = -1,
    ):
        super().__init__()
        self.default_id = int(default_id)

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

        num_seen = len(seen_ids)
        self.depth = num_seen + 1
        self.oov_bucket = num_seen

        self._seen_ids = seen_ids
        self._id_to_bucket = {int(id_): bucket for bucket, id_ in enumerate(seen_ids)}

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

        oh = F.one_hot(bucket_idx, num_classes=self.depth).float()
        return self.dense(oh)

    @property
    def weight_matrix(self) -> torch.Tensor:
        return self.dense.weight.T


class MultiOneHotDenseEncoderTorch(nn.Module):
    def __init__(
        self,
        feature_indexes: List[int],
        training_ids: List[List[int]],
        output_dims: List[int],
    ):
        super().__init__()
        self.feature_indexes = feature_indexes
        self.training_ids = training_ids
        self.output_dims = output_dims

        self.singleIdEncoders = nn.ModuleList([
            OneHotDenseEncoderTorch(
                train_ids=train_ids,
                output_dim=output_dim,
            )
            for train_ids, output_dim in zip(training_ids, output_dims)
        ])

        self.added_buffer_size = sum(output_dims) - len(feature_indexes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_size = int(input_tensor.shape[1])

        indexes_to_keep = [
            i for i in range(input_size) if i not in self.feature_indexes
        ]

        gather_output = input_tensor[:, indexes_to_keep]

        encoded_features = [
            encoder(input_tensor[:, index])
            for index, encoder in zip(self.feature_indexes, self.singleIdEncoders)
        ]

        return torch.cat([gather_output] + encoded_features, dim=-1)


###############################################################################
# DataBuffer Classes
###############################################################################

class DataBuffer:
    def __init__(self, shape, op=None, inputs=None):
        self._shape = shape
        self.op = op
        self.inputs = inputs if inputs is not None else []

    def __repr__(self):
        return f"DataBuffer(shape={self._shape})"

    @property
    def os(self):
        return self._shape[-1]

    def __getitem__(self, key):
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
    def __init__(self, shape_or_os, name=None):
        if isinstance(shape_or_os, int):
            shape = (shape_or_os,)
        else:
            shape = shape_or_os
        super().__init__(shape, op=None, inputs=[])
        self.shape = shape

    def __repr__(self):
        return f"InputBuffer(shape={self.shape})"

    def __call__(self, *args, **kwargs):
        return self


###############################################################################
# ComputationOp Base Class and Ops
###############################################################################

class ComputationOp(ABC):
    def __init__(self):
        self.torch_module = None

    @abstractmethod
    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        raise NotImplementedError("Subclasses must implement __call__.")

    @abstractmethod
    def compile_instructions(
        self, input_indices, weights_visited, model_structure
    ) -> int:
        raise NotImplementedError("Subclasses must implement compile_instructions.")

    def forward_op(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward_op.")


class ActivationComputation(ComputationOp):
    def __init__(self, activation, in_place=False, name: Optional[str] = None):
        super().__init__()
        self.in_place = in_place
        self.activation = activation.upper()
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        output_shape = inputs[0]._shape
        return DataBuffer(output_shape, op=self, inputs=inputs)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_activation(x, self.activation)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        if len(input_indices) != 1:
            raise ValueError("ActivationComputation expects exactly one input.")

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
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("ReduceSum expects exactly one input.")
        return DataBuffer((1,), op=self, inputs=inputs)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1, keepdim=True)

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
    def __init__(self, output_size, activation=None, use_bias=True, name=None):
        super().__init__()
        self.input_size = None
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        self.name = name
        self.torch_module = None

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("Dense expects exactly one input in the list.")
        input_shape = inputs[0]._shape
        if self.input_size is None:
            self.input_size = input_shape[-1]
            self.torch_module = nn.Linear(
                self.input_size, self.output_size, bias=self.use_bias
            )
        return DataBuffer((self.output_size,), op=self, inputs=inputs)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        out = self.torch_module(x)
        if self.activation is not None:
            out = _apply_activation(out, self.activation)
        return out

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        if len(input_indices) != 1:
            raise ValueError("Dense.compile_instructions expects one input index.")

        wv = weights_visited["weights"]
        if id(self) not in wv:
            wv[id(self)] = len(model_structure["weights"])

            weight = self.torch_module.weight.detach().cpu().numpy()
            model_structure["weights"].append(weight.tolist())

            if self.use_bias:
                bias = self.torch_module.bias.detach().cpu().numpy()
                model_structure["bias"].append(bias.tolist())
            else:
                bias = [0.0] * self.output_size
                model_structure["bias"].append(bias)

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
    def __init__(self, indexes, name=None):
        super().__init__()
        self.indexes = indexes
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        return DataBuffer((len(self.indexes),), op=self, inputs=inputs)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.indexes]

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
    def __init__(self, axis=-1, name=None):
        super().__init__()
        self.axis = axis

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        total = sum(inp.os for inp in inputs)
        return DataBuffer((total,), op=self, inputs=inputs)

    def forward_op(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        return torch.cat(input_tensors, dim=-1)

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
    def __init__(
        self,
        feature_indexes: List[int],
        training_ids: List[List[int]],
        output_dims: List[int],
        name: Optional[str] = None,
    ):
        super().__init__()
        self.layer = MultiOneHotDenseEncoderTorch(
            feature_indexes=feature_indexes,
            training_ids=training_ids,
            output_dims=output_dims,
        )

        self.feature_indexes = feature_indexes
        self.training_ids = training_ids
        self.output_dims = output_dims
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("MultiIdEmbeddings expects exactly one input buffer.")

        input_size = inputs[0].os
        output_size = input_size + self.layer.added_buffer_size
        return DataBuffer((output_size,), op=self, inputs=inputs)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

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

        maps_cache = weights_visited["maps"]

        for encoder_id, encoder in enumerate(self.layer.singleIdEncoders):
            weight_matrix = encoder.weight_matrix.detach().cpu().numpy()
            map_dict: Dict[int, List[float]] = {}

            for bucket_idx, real_id in enumerate(self.training_ids[encoder_id]):
                vector = weight_matrix[bucket_idx].astype(np.float32).tolist()
                map_dict[int(real_id)] = vector

            default_vector = weight_matrix[-1].astype(np.float32).tolist()
            maps_cache[id(self)] = len(model_structure["maps"])
            model_structure["maps"].append(map_dict)

            map_index = maps_cache[id(self)]

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


class SingleIdEmbeddings(ComputationOp):
    def __init__(
        self,
        train_ids: List[int],
        output_dim: int,
        internal_input_index: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__()
        default_id = -1
        self.layer = OneHotDenseEncoderTorch(
            train_ids=train_ids, output_dim=output_dim, default_id=default_id
        )
        self.output_dim = int(output_dim)
        self.default_id = int(default_id)
        self.train_ids = list(dict.fromkeys(train_ids))
        self.internal_input_index = internal_input_index
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("SingleIdEmbeddings expects exactly one input buffer.")

        return DataBuffer((self.output_dim,), op=self, inputs=inputs)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

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

        maps_cache = weights_visited["maps"]
        weight_matrix = self.layer.weight_matrix.detach().cpu().numpy()
        map_dict: Dict[int, List[float]] = {}

        for bucket_idx, real_id in enumerate(self.train_ids):
            vector = weight_matrix[bucket_idx].astype(np.float32).tolist()
            map_dict[int(real_id)] = vector

        default_vector = weight_matrix[-1].astype(np.float32).tolist()
        maps_cache[id(self)] = len(model_structure["maps"])
        model_structure["maps"].append(map_dict)

        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(self.output_dim)

        map_index = maps_cache[id(self)]

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
    def __init__(self, in_place=False, center=True, scale=True, epsilon=1e-3, name=None):
        super().__init__()
        self.in_place = in_place
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.name = name
        self._bn = None

    def _ensure_bn(self, num_features):
        if self._bn is None:
            self._bn = nn.BatchNorm1d(
                num_features,
                eps=self.epsilon,
                affine=(self.center or self.scale),
            )
            if self._bn.weight is not None and not self.scale:
                self._bn.weight.requires_grad_(False)
                self._bn.weight.fill_(1.0)
            if self._bn.bias is not None and not self.center:
                self._bn.bias.requires_grad_(False)
                self._bn.bias.fill_(0.0)

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        num_features = inputs[0].os
        self._ensure_bn(num_features)
        return DataBuffer((num_features,), op=self, inputs=inputs)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        return self._bn(x)

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        bn = self._bn

        gamma = bn.weight.detach().cpu().numpy() if bn.weight is not None else np.ones(bn.num_features)
        beta = bn.bias.detach().cpu().numpy() if bn.bias is not None else np.zeros(bn.num_features)
        moving_mean = bn.running_mean.detach().cpu().numpy()
        moving_variance = bn.running_var.detach().cpu().numpy()

        if not self.scale:
            gamma = np.ones_like(moving_mean)
        if not self.center:
            beta = np.zeros_like(gamma)

        epsilon = self.epsilon
        std = gamma / np.sqrt(moving_variance + epsilon)
        center_param = -moving_mean

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
            model_structure["parameters"].append(center_param.tolist())
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
    def __init__(self, scaling_vector, in_place=False, name=None):
        super().__init__()
        arr = np.asarray(scaling_vector, dtype=np.float32)
        self._is_scalar = arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1)
        if self._is_scalar:
            self._scalar_value = float(arr.flat[0])
            self.scaling_vector = None
        else:
            self._scalar_value = None
            self.scaling_vector = arr
        self.in_place = in_place
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("ScaleVectorized expects exactly one input.")
        return DataBuffer(inputs[0]._shape, op=self, inputs=inputs)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_scalar:
            return x * self._scalar_value
        else:
            t = torch.tensor(self.scaling_vector, dtype=x.dtype, device=x.device)
            return x * t

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
            if self._is_scalar:
                buffer_size = model_structure["buffer_sizes"][input_indices[0]]
                expanded_vector = [self._scalar_value] * buffer_size
                model_structure["parameters"].append(expanded_vector)
            else:
                model_structure["parameters"].append(self.scaling_vector.tolist())

        instr = {
            "type": "MUL_ELEMENTWISE",
            "input": target_index,
            "parameters": pw[id(self)],
        }
        model_structure["instructions"].append(instr)
        return target_index


class ShiftVectorized(ComputationOp):
    def __init__(self, shift_vector, in_place=False, name=None):
        super().__init__()
        arr = np.asarray(shift_vector, dtype=np.float32)
        self._is_scalar = arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1)
        if self._is_scalar:
            self._scalar_value = float(arr.flat[0])
            self.shift_vector = None
        else:
            self._scalar_value = None
            self.shift_vector = arr
        self.in_place = in_place
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("ShiftVectorized expects exactly one input.")
        return DataBuffer(inputs[0]._shape, op=self, inputs=inputs)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_scalar:
            return x + self._scalar_value
        else:
            t = torch.tensor(self.shift_vector, dtype=x.dtype, device=x.device)
            return x + t

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
            if self._is_scalar:
                buffer_size = model_structure["buffer_sizes"][input_indices[0]]
                expanded_vector = [self._scalar_value] * buffer_size
                model_structure["parameters"].append(expanded_vector)
            else:
                model_structure["parameters"].append(self.shift_vector.tolist())

        instr = {
            "type": "ADD_ELEMENTWISE",
            "input": target_index,
            "parameters": pw[id(self)],
        }
        model_structure["instructions"].append(instr)
        return target_index


class Attention(ComputationOp):
    def __init__(self, name=None):
        super().__init__()
        self.a = None
        self.b = None
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("Attention expects two inputs: [target, key]")
        target, key = inputs

        if self.torch_module is None:
            self.a = target.os
            self.b = key.os
            self.torch_module = nn.Linear(self.b, self.a)

        return DataBuffer((self.a,), op=self, inputs=inputs)

    def forward_op(self, target: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        softmaxed = F.softmax(self.torch_module(key), dim=-1)
        return target * softmaxed

    def compile_instructions(self, input_indices, weights_visited, model_structure):
        wv = weights_visited["weights"]
        if id(self) not in wv:
            wv[id(self)] = len(model_structure["weights"])
            weight = self.torch_module.weight.detach().cpu().numpy()
            bias = self.torch_module.bias.detach().cpu().numpy()
            model_structure["weights"].append(weight.tolist())
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
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        out_size = inputs[0].os
        return DataBuffer((out_size,), op=self, inputs=inputs)

    def forward_op(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(input_tensors, dim=0)
        return torch.sum(stacked, dim=0)

    def compile_instructions(
        self, input_indices, weights_visited, model_structure
    ) -> int:
        if not input_indices:
            raise ValueError("Add expects at least one input.")
        out_size = model_structure["buffer_sizes"][input_indices[0]]
        for idx in input_indices:
            if model_structure["buffer_sizes"][idx] != out_size:
                raise ValueError("All inputs must have the same size for addition.")
        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(out_size)
        instr = {
            "type": "ADD_ELEMENTWISE_BUFFERS",
            "input": input_indices,
            "output": output_index,
        }
        model_structure["instructions"].append(instr)
        return output_index


class Multiply(ComputationOp):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name

    def __call__(self, inputs):
        if not isinstance(inputs, list) or len(inputs) < 2:
            raise ValueError("Multiply expects at least two inputs.")
        out_size = inputs[0].os
        return DataBuffer((out_size,), op=self, inputs=inputs)

    def forward_op(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(input_tensors, dim=0)
        return torch.prod(stacked, dim=0)

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


class MultiplyHeads(ComputationOp):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.head_dim = None

    def __call__(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("MultiplyHeads expects exactly two inputs.")
        data_buffer, heads_buffer = inputs
        data_size = data_buffer.os
        heads_size = heads_buffer.os
        if data_size % heads_size != 0:
            raise ValueError(
                f"Data buffer size ({data_size}) must be divisible by heads buffer size ({heads_size})."
            )
        self.head_dim = data_size // heads_size
        return DataBuffer((data_size,), op=self, inputs=inputs)

    def forward_op(self, data: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        expanded = torch.repeat_interleave(heads, self.head_dim, dim=-1)
        return data * expanded

    def compile_instructions(
        self, input_indices, weights_visited, model_structure
    ) -> int:
        if len(input_indices) != 2:
            raise ValueError("MultiplyHeads expects exactly two inputs.")
        data_idx, heads_idx = input_indices
        data_size = model_structure["buffer_sizes"][data_idx]
        heads_size = model_structure["buffer_sizes"][heads_idx]
        if data_size % heads_size != 0:
            raise ValueError(
                f"Data buffer size ({data_size}) must be divisible by heads buffer size ({heads_size})."
            )
        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(data_size)
        instr = {
            "type": "MULTIPLY_BUFFER_HEADS",
            "input": [data_idx, heads_idx],
            "output": output_index,
        }
        model_structure["instructions"].append(instr)
        return output_index


class AddHeads(ComputationOp):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.head_dim = None

    def __call__(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("AddHeads expects exactly two inputs.")
        data_buffer, heads_buffer = inputs
        data_size = data_buffer.os
        heads_size = heads_buffer.os
        if data_size % heads_size != 0:
            raise ValueError(
                f"Data buffer size ({data_size}) must be divisible by heads buffer size ({heads_size})."
            )
        self.head_dim = data_size // heads_size
        return DataBuffer((data_size,), op=self, inputs=inputs)

    def forward_op(self, data: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        expanded = torch.repeat_interleave(heads, self.head_dim, dim=-1)
        return data + expanded

    def compile_instructions(
        self, input_indices, weights_visited, model_structure
    ) -> int:
        if len(input_indices) != 2:
            raise ValueError("AddHeads expects exactly two inputs.")
        data_idx, heads_idx = input_indices
        data_size = model_structure["buffer_sizes"][data_idx]
        heads_size = model_structure["buffer_sizes"][heads_idx]
        if data_size % heads_size != 0:
            raise ValueError(
                f"Data buffer size ({data_size}) must be divisible by heads buffer size ({heads_size})."
            )
        output_index = len(model_structure["buffer_sizes"])
        model_structure["buffer_sizes"].append(data_size)
        instr = {
            "type": "ADD_BUFFER_HEADS",
            "input": [data_idx, heads_idx],
            "output": output_index,
        }
        model_structure["instructions"].append(instr)
        return output_index


###############################################################################
# _InternalModule and ModelGraph
###############################################################################

class _InternalModule(nn.Module):
    def __init__(self, input_buffers, output_buffer):
        super().__init__()
        self._input_buffers = input_buffers if isinstance(input_buffers, list) else [input_buffers]
        self._output_buffer = output_buffer

        self._topo_order = []
        self._buffer_to_idx = {}
        self._modules_dict = nn.ModuleDict()

        visited = set()
        self._build_topo(output_buffer, visited)

    def _build_topo(self, buffer, visited):
        bid = id(buffer)
        if bid in visited:
            return
        visited.add(bid)

        if isinstance(buffer, InputBuffer):
            return

        for inp in buffer.inputs:
            self._build_topo(inp, visited)

        if buffer.op is not None:
            self._topo_order.append(buffer)
            op = buffer.op
            self._register_op_modules(op)

    def _register_op_modules(self, op):
        op_id = str(id(op))
        if op_id in self._modules_dict:
            return
        if isinstance(op, Dense) and op.torch_module is not None:
            self._modules_dict[op_id] = op.torch_module
        elif isinstance(op, Attention) and op.torch_module is not None:
            self._modules_dict[op_id] = op.torch_module
        elif isinstance(op, NormalizationComputation) and op._bn is not None:
            self._modules_dict[op_id] = op._bn
        elif isinstance(op, SingleIdEmbeddings):
            self._modules_dict[op_id] = op.layer
        elif isinstance(op, MultiIdEmbeddings):
            self._modules_dict[op_id] = op.layer
        elif isinstance(op, ModelGraph):
            self._modules_dict[op_id] = op._torch_module

    def forward(self, *inputs):
        tensor_map = {}
        for i, buf in enumerate(self._input_buffers):
            tensor_map[id(buf)] = inputs[i]

        for buffer in self._topo_order:
            op = buffer.op

            if isinstance(op, ModelGraph):
                inp_tensors = [tensor_map[id(b)] for b in buffer.inputs]
                tensor_map[id(buffer)] = op._torch_module(*inp_tensors)
            else:
                inp_tensors = [tensor_map[id(b)] for b in buffer.inputs]
                tensor_map[id(buffer)] = op.forward_op(*inp_tensors)

        return tensor_map[id(self._output_buffer)]


class ModelGraph(ComputationOp):
    def __init__(self, input_buffers, output_buffer: DataBuffer, name=None, device=None):
        super().__init__()
        if not isinstance(input_buffers, list):
            input_buffers = [input_buffers]
        self.input_buffers = input_buffers
        self.output_buffer = output_buffer
        self._device = device or _default_device()
        self._torch_module = _InternalModule(input_buffers, output_buffer).to(self._device)
        self._optimizer = None
        self._loss_fn = None

    @property
    def os(self):
        return self.output_buffer.os

    def get_torch(self):
        return self._torch_module

    def get_module(self):
        return self._torch_module

    def compile(self, optimizer="adam", loss="mse", lr=0.001, metrics=None, **kwargs):
        self._metrics = metrics or []
        opt_map = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }
        loss_map = {
            "mse": nn.MSELoss,
            "binary_crossentropy": nn.BCELoss,
            "bce": nn.BCELoss,
            "crossentropy": nn.CrossEntropyLoss,
            "cross_entropy": nn.CrossEntropyLoss,
            "mae": nn.L1Loss,
            "l1": nn.L1Loss,
        }

        if isinstance(optimizer, str):
            opt_cls = opt_map.get(optimizer.lower())
            if opt_cls is None:
                raise ValueError(f"Unknown optimizer: {optimizer}")
            self._optimizer = opt_cls(self._torch_module.parameters(), lr=lr, **kwargs)
        else:
            self._optimizer = optimizer

        if isinstance(loss, str):
            loss_cls = loss_map.get(loss.lower())
            if loss_cls is None:
                raise ValueError(f"Unknown loss: {loss}")
            self._loss_fn = loss_cls()
        else:
            self._loss_fn = loss

    def fit(self, x, y, epochs=1, batch_size=32, verbose=1, shuffle=True, sample_weight=None):
        device = self._device
        self._torch_module.train()

        if isinstance(x, list):
            x_tensors = [torch.tensor(np.asarray(xi), dtype=torch.float32, device=device) for xi in x]
            n_samples = x_tensors[0].shape[0]
        else:
            x_tensors = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
            n_samples = x_tensors.shape[0]

        y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32, device=device)

        if sample_weight is not None:
            w_tensor = torch.tensor(np.asarray(sample_weight), dtype=torch.float32, device=device)
        else:
            w_tensor = None

        for epoch in range(epochs):
            if shuffle:
                perm = torch.randperm(n_samples, device=device)
                if isinstance(x_tensors, list):
                    x_tensors_epoch = [xt[perm] for xt in x_tensors]
                else:
                    x_tensors_epoch = x_tensors[perm]
                y_epoch = y_tensor[perm]
                w_epoch = w_tensor[perm] if w_tensor is not None else None
            else:
                x_tensors_epoch = x_tensors
                y_epoch = y_tensor
                w_epoch = w_tensor

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                if isinstance(x_tensors_epoch, list):
                    x_batch = [xt[start:end] for xt in x_tensors_epoch]
                else:
                    x_batch = x_tensors_epoch[start:end]
                y_batch = y_epoch[start:end]

                self._optimizer.zero_grad()
                if isinstance(x_batch, list):
                    pred = self._torch_module(*x_batch)
                else:
                    pred = self._torch_module(x_batch)

                if w_epoch is not None:
                    w_batch = w_epoch[start:end]
                    per_sample_loss = (pred - y_batch).pow(2).mean(dim=-1)
                    loss = (per_sample_loss * w_batch).mean()
                else:
                    loss = self._loss_fn(pred, y_batch)

                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if verbose:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    def predict(self, x, verbose=0):
        device = self._device
        self._torch_module.eval()
        with torch.no_grad():
            if isinstance(x, list):
                x_tensors = [torch.tensor(np.asarray(xi), dtype=torch.float32, device=device) for xi in x]
                pred = self._torch_module(*x_tensors)
            else:
                x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
                pred = self._torch_module(x_tensor)
        return pred.cpu().numpy()

    def predict_on_batch(self, x):
        return self.predict(x)

    def summary(self):
        print(self._torch_module)

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        output_shape = self.output_buffer._shape
        return DataBuffer(output_shape, op=self, inputs=inputs)

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
                model_structure["buffer_sizes"].append(input_buffer.os)
                visited[id(input_buffer)] = idx

        def traverse(buffer: DataBuffer):
            if id(buffer) in visited:
                return visited[id(buffer)]
            if buffer.op is None:
                idx = len(model_structure["buffer_sizes"])
                model_structure["buffer_sizes"].append(buffer.os)
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


def generate_validation_data(
    features: list[str],
    model: ModelGraph,
    means=None,
    stds=None,
):
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
    n_samples, n_cols = y_data.shape
    results = []

    for col in range(n_cols):
        pred_col = predictions[:, col]
        y_col = y_data[:, col]

        tau, p_value = kendalltau(pred_col, y_col)
        if np.isnan(tau):
            tau = 0.0

        results.append(tau)

    return results if len(results) > 1 else results[0]


def score_selection(model, x_data, y_data):
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


def validate_torch_model(model, validation_data):
    x_val = np.array(validation_data["inputs"])
    y_expected = np.array(validation_data["expected_outputs"])
    y_pred = model.predict(x_val)
    if np.allclose(y_expected, y_pred, atol=1e-6):
        print("PyTorch model validation successful: predictions match expected outputs.")
    else:
        print("PyTorch model validation failed.")
        print("Expected outputs:", y_expected)
        print("Predictions:", y_pred)
        raise AssertionError("PyTorch model validation failed.")


validate_model = validate_torch_model
