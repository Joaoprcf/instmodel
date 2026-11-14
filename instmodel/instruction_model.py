import json
import operator
import os

import numpy as np
from scipy.special import erf


def load_model_from_label(model_location):
    if not os.path.exists(model_location):
        raise ValueError(f"Model not found at {model_location}")
    with open(model_location, "r") as f:
        return json.load(f)


def generate_features(model_label, dataset, output_features):
    model_json = load_model_from_label(model_label)
    return generate_features_IO(model_label, dataset, model_json["features"], output_features)


def generate_features_IO(model_label, dataset, input_features, output_features):
    """Returns a copy of the DataFrame with output_features set to the model's outputs."""
    model_json = load_model_from_label(model_label)
    *_, output = instruction_model_inference(model_json, dataset[input_features].to_numpy())
    dataset_new = dataset.copy()
    dataset_new[output_features] = output
    return dataset_new


def _softmax(x):
    axis = 0 if x.ndim == 1 else 1
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _log_positive(x, log):
    result = np.zeros_like(x)
    mask = x > 0
    result[mask] = log(x[mask] + 1)
    return result


# Same switch points as TensorFlow's float32 softplus: log(eps) + 2.
_SOFTPLUS_THRESHOLD = np.float32(np.log(np.finfo(np.float32).eps) + 2)

_ACTIVATIONS = {
    "RELU": lambda x: np.maximum(0, x),
    "SIGMOID": lambda x: np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))),
    "SOFTMAX": _softmax,
    "TANH": np.tanh,
    "SQRT": lambda x: np.where(x > 0, np.sqrt(x), 0),
    "LOG": lambda x: _log_positive(x, np.log),
    "LOG10": lambda x: _log_positive(x, np.log10),
    "INVERSE": lambda x: 1 - x,
    "GELU": lambda x: x * 0.5 * (1 + erf(x / np.sqrt(2))),
    "SOFTPLUS": lambda x: np.where(
        x > -_SOFTPLUS_THRESHOLD,
        x,
        np.where(x < _SOFTPLUS_THRESHOLD, np.exp(x), np.log1p(np.exp(x))),
    ),
    "EXP": lambda x: np.exp(np.clip(x, -88, 88)),  # stays finite in float32
    "SIGN": np.sign,
}


def apply_activation(instruction, buffers, output_index):
    """Applies instruction["activation"] (if any) in place to buffers[output_index]."""
    act = instruction.get("activation")
    if act is None:
        return
    fn = _ACTIVATIONS.get(act)
    if fn is None:
        raise ValueError(f"Unexpected activation: {act}")
    buffers[output_index] = fn(buffers[output_index])


def _check_index(value, count, what):
    try:
        index = operator.index(value)
    except TypeError:
        raise ValueError(f"{what} index must be an integer, got {value!r}") from None
    if not 0 <= index < count:
        raise ValueError(f"{what} index {index} out of range [0, {count})")
    return index


def _check_span(start, length, capacity, what):
    start = _check_index(start, capacity + 1, f"{what} offset")
    if start + length > capacity:
        raise ValueError(f"{what} writes columns [{start}, {start + length}) past buffer size {capacity}")


def _validate_references(model):
    """Rejects out-of-range or negative references, which NumPy would otherwise wrap or misread."""
    sizes = model["buffer_sizes"]
    n_buffers = len(sizes)
    counts = {
        "weights": len(model["weights"]),
        "parameters": len(model["parameters"]),
        "parameters_min": len(model["parameters"]),
        "parameters_max": len(model["parameters"]),
        "map": len(model.get("maps", [])),
    }
    for instr in model["instructions"]:
        inputs = instr.get("input")
        for value in inputs if isinstance(inputs, list) else [inputs]:
            _check_index(value, n_buffers, "buffer")
        for key in ("output", "key"):
            if key in instr:
                _check_index(instr[key], n_buffers, "buffer")
        for key, count in counts.items():
            if key in instr:
                _check_index(instr[key], count, key)
        if instr["type"] == "COPY_MASKED":
            if len(instr["indexes"]) > sizes[instr["output"]]:
                raise ValueError("COPY_MASKED selects more columns than its output buffer holds")
            for value in instr["indexes"]:
                _check_index(value, sizes[instr["input"]], "COPY_MASKED column")
        elif instr["type"] == "COPY":
            _check_span(instr["internal_index"], sizes[instr["input"]], sizes[instr["output"]], "COPY")
        elif instr["type"] == "MAP_TRANSFORM":
            _check_index(instr["internal_input_index"], sizes[instr["input"]], "MAP_TRANSFORM column")
            _check_span(
                instr["internal_output_index"], instr["size"], sizes[instr["output"]], "MAP_TRANSFORM"
            )


def _output_buffer(buffers, buffer_sizes, index, rows):
    if buffers[index] is None:
        buffers[index] = np.zeros((rows, buffer_sizes[index]), dtype=np.float32)
    return buffers[index]


_BUFFER_REDUCERS = {"ADD_ELEMENTWISE_BUFFERS": np.sum, "MULTIPLY_ELEMENTWISE_BUFFERS": np.prod}
_HEAD_COMBINERS = {"ADD_BUFFER_HEADS": np.add, "MULTIPLY_BUFFER_HEADS": np.multiply}


def instruction_model_inference(model, input_data):
    """Runs the model on a 2D array (split across the leading buffers) or a list of per-buffer arrays.

    Returns every buffer; the last one is the model output.
    """
    _validate_references(model)
    weights = [np.array(w, dtype=np.float32) for w in model["weights"]]
    bias = [np.array(b, dtype=np.float32) for b in model["bias"]]
    parameters = [np.array(p, dtype=np.float32) for p in model["parameters"]]
    buffer_sizes = model["buffer_sizes"]
    buffers = [None] * len(buffer_sizes)

    if isinstance(input_data, list):
        for i, data in enumerate(input_data):
            buffers[i] = data.copy()
            if buffers[i].shape[1] != buffer_sizes[i]:
                raise ValueError(
                    f"Input data shape does not match the expected shape. Expected {buffer_sizes[i]} but got {buffers[i].shape[1]}"
                )
    else:
        total_features = input_data.shape[1]
        accumulated_capacities = []
        accumulated = 0
        for size in buffer_sizes:
            accumulated += size
            accumulated_capacities.append(accumulated)
            if accumulated >= total_features:
                break
        if not accumulated_capacities or accumulated != total_features:
            raise ValueError(
                f"Invalid input feature size: expected one of the cumulative capacities {accumulated_capacities} but got {total_features}"
            )
        start = 0
        for i, size in enumerate(buffer_sizes[: len(accumulated_capacities)]):
            buffers[i] = input_data[:, start : start + size].copy()
            start += size

    for instruction in model["instructions"]:
        kind = instruction["type"]
        src = instruction["input"]
        if kind == "ACTIVATION":
            apply_activation(instruction, buffers, src)
            continue
        if kind in _BUFFER_REDUCERS:
            stacked = np.stack([buffers[i] for i in src], axis=0)
            buffers[instruction["output"]] = np.asarray(
                _BUFFER_REDUCERS[kind](stacked, axis=0), dtype=np.float32
            )
            continue
        if kind in _HEAD_COMBINERS:
            data, heads = buffers[src[0]], buffers[src[1]]
            expanded = np.repeat(heads, repeats=data.shape[1] // heads.shape[1], axis=1)
            buffers[instruction["output"]] = np.asarray(
                _HEAD_COMBINERS[kind](data, expanded), dtype=np.float32
            )
            continue
        if kind == "REDUCE_SUM":
            buffers[instruction["output"]] = np.asarray(
                np.sum(buffers[src], axis=-1, keepdims=True), dtype=np.float32
            )
            continue

        x = buffers[src]
        if kind == "COPY_MASKED":
            indexes = instruction["indexes"]
            out = _output_buffer(buffers, buffer_sizes, instruction["output"], x.shape[0])
            out[:, : len(indexes)] = x[:, indexes]
        elif kind == "COPY":
            start = instruction["internal_index"]
            out = _output_buffer(buffers, buffer_sizes, instruction["output"], x.shape[0])
            out[:, start : start + buffer_sizes[src]] = x
        elif kind == "MAP_TRANSFORM":
            size = instruction["size"]
            default = instruction["default"]
            map_data = {int(k): v for k, v in model["maps"][instruction["map"]].items()}
            col = instruction["internal_input_index"]
            start = instruction["internal_output_index"]
            out = _output_buffer(buffers, buffer_sizes, instruction["output"], x.shape[0])
            out[:, start : start + size] = np.array(
                [np.array(map_data.get(i.item(), default)) for i in x[:, col]]
            ).reshape(-1, size)
        elif kind == "DOT":
            w = instruction["weights"]
            buffers[instruction["output"]] = x @ weights[w].T + bias[w]
            apply_activation(instruction, buffers, instruction["output"])
        elif kind == "ATTENTION":
            w = instruction["weights"]
            scores = buffers[instruction["key"]] @ weights[w].T + bias[w]
            buffers[instruction["output"]] = _softmax(scores) * x
        elif kind == "ADD_ELEMENTWISE":
            buffers[src] = x + parameters[instruction["parameters"]]
        elif kind == "MUL_ELEMENTWISE":
            buffers[src] = x * parameters[instruction["parameters"]]
        elif kind == "CLIP_ELEMENTWISE":
            if "parameters_min" in instruction:
                x = np.maximum(x, parameters[instruction["parameters_min"]])
            if "parameters_max" in instruction:
                x = np.minimum(x, parameters[instruction["parameters_max"]])
            buffers[src] = x
        else:
            raise ValueError(f"Unknown instruction type: {kind}")
        buffers[src] = np.asarray(buffers[src], dtype=np.float32)

    return buffers


def validate_instruction_model(model):
    """Raises AssertionError unless inference on validation_data matches expected_outputs (atol 1e-6)."""
    expected = np.array(model["validation_data"]["expected_outputs"])
    actual = instruction_model_inference(model, np.array(model["validation_data"]["inputs"]))[-1]
    if not np.allclose(expected, actual, atol=1e-6):
        print("output_data_all shape:", expected.shape)
        print("output_inference shape:", actual.shape)
        print("MAX ERROR: ", np.max(np.abs(expected - actual)))
        print("MEAN ERROR: ", np.mean(np.abs(expected - actual)))
        print("All ERROR: ", np.abs(expected - actual))
        raise AssertionError("Instruction model output does not match validation_data.")


def concatenate_instruction_models(models, start_from_first=False):
    """Chains models: each model's first buffer is the previous model's last buffer."""
    concatenated_buffer_sizes = models[0]["buffer_sizes"][:]
    for model in models[1:]:
        if concatenated_buffer_sizes[-1] != model["buffer_sizes"][0]:
            raise AssertionError("Last buffer of a model must match the first buffer of the next.")
        concatenated_buffer_sizes += model["buffer_sizes"][1:]

    if start_from_first:
        new_model = models[0]
        offset = len(new_model["buffer_sizes"]) - 1
        new_model["buffer_sizes"] = concatenated_buffer_sizes
    else:
        new_model = {
            "features": models[0]["features"],
            "buffer_sizes": concatenated_buffer_sizes,
            "instructions": [],
            "weights": [],
            "bias": [],
            "parameters": [],
            "maps": [],
        }
        offset = 0

    for model in models[int(start_from_first) :]:
        shifts = {
            "weights": len(new_model["weights"]),
            "parameters": len(new_model["parameters"]),
            "parameters_min": len(new_model["parameters"]),
            "parameters_max": len(new_model["parameters"]),
            "map": len(new_model["maps"]),
            "output": offset,
            "key": offset,
        }
        for instruction in model["instructions"]:
            new_instruction = instruction.copy()
            for key, shift in shifts.items():
                if key in new_instruction:
                    new_instruction[key] += shift
            if "input" in new_instruction:
                inputs = new_instruction["input"]
                new_instruction["input"] = (
                    [x + offset for x in inputs] if isinstance(inputs, list) else inputs + offset
                )
            new_model["instructions"].append(new_instruction)

        for key in ("weights", "bias", "parameters", "maps"):
            new_model[key] += model[key]
        offset += len(model["buffer_sizes"]) - 1
    return new_model


def create_instructions_model_from_transformation_list(features, transformation_list):
    """Builds a feature-preprocessing model from {"delete": f} and {"from", "to", "map", "size", "default"} items.

    Returns (instruction model, output feature names).
    """
    new_size = (
        len(features)
        + sum(i["size"] for i in transformation_list if "to" in i)
        - sum(1 for i in transformation_list if "delete" in i)
    )
    transformation_list = sorted(transformation_list, key=lambda x: "delete" in x, reverse=True)

    new_features = features.copy()
    keep_indexes = [*range(len(features))]
    for transformation in transformation_list:
        if "delete" in transformation:
            new_features.remove(transformation["delete"])
            keep_indexes.remove(features.index(transformation["delete"]))
        elif "to" in transformation:
            new_features.extend(transformation["to"])

    instructions = [{"type": "COPY_MASKED", "input": 0, "output": 1, "indexes": keep_indexes}]
    maps = []
    for transformation in transformation_list:
        if "to" in transformation:
            instructions.append(
                {
                    "type": "MAP_TRANSFORM",
                    "input": 0,
                    "output": 1,
                    "internal_input_index": features.index(transformation["from"]),
                    "internal_output_index": new_features.index(transformation["to"][0]),
                    "map": len(maps),
                    "size": transformation["size"],
                    "default": transformation["default"],
                }
            )
            maps.append(transformation["map"])

    return {
        "features": features,
        "buffer_sizes": [len(features), new_size],
        "instructions": instructions,
        "weights": [],
        "bias": [],
        "parameters": [],
        "maps": maps,
    }, new_features
