import numpy as np
import json
import os
import pandas as pd


def load_model_from_label(model_location):
    if os.path.exists(model_location):
        with open(model_location, "r") as f:
            model_json = json.load(f)
    else:
        raise ValueError(f"Model not found at {model_location}")
    return model_json


def generate_features(model_label, dataset, output_features):
    model_json = load_model_from_label(model_label)

    return generate_features_IO(
        model_label, dataset, model_json["features"], output_features
    )


def generate_features_IO(model_label, dataset, input_features, output_features):
    # before going to s3, check if the model is already in the local cache in folder models

    model_json = load_model_from_label(model_label)

    *_, output = instruction_model_inference(
        model_json, dataset[input_features].to_numpy()
    )

    dataset_new = dataset.copy()
    dataset_new[output_features] = output

    return dataset_new


def apply_activation(instruction, buffers, output_index):
    """
    Applies the activation function specified in the instruction to the buffer at output_index.

    Parameters:
        instruction (dict): A dictionary that may contain an "activation" key.
        buffers (dict or list): A container of NumPy arrays.
        output_index: The key or index for the output buffer within `buffers`.
    """
    act = instruction.get("activation")
    if act is None:
        # No activation specified
        return

    x = buffers[output_index]

    if act == "RELU":
        buffers[output_index] = np.maximum(0, x)

    elif act == "SIGMOID":
        # Using a numerically stable sigmoid computation
        buffers[output_index] = np.where(
            x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))
        )

    elif act == "SOFTMAX":
        # Determine axis based on array dimensions
        if x.ndim == 1:
            # For a 1D vector, subtract the max for numerical stability
            shifted_x = x - np.max(x)
            exp_x = np.exp(shifted_x)
            buffers[output_index] = exp_x / np.sum(exp_x)
        else:
            # For 2D (or higher) arrays, assume softmax is computed along axis=1
            shifted_x = x - np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(shifted_x)
            buffers[output_index] = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    elif act == "TANH":
        buffers[output_index] = np.tanh(x)

    elif act == "SQRT":
        # Compute sqrt(x) for x > 0, else 0
        buffers[output_index] = np.where(x > 0, np.sqrt(x), 0)

    elif act == "LOG":
        # Compute log(x+1) for x > 0, else 0
        result = np.zeros_like(x)
        mask = x > 0
        result[mask] = np.log(x[mask] + 1)
        buffers[output_index] = result

    elif act == "LOG10":
        # Compute log10(x+1) for x > 0, else 0
        result = np.zeros_like(x)
        mask = x > 0
        result[mask] = np.log10(x[mask] + 1)
        buffers[output_index] = result

    elif act == "INVERSE":
        # Compute 1 - x
        buffers[output_index] = 1 - x

    else:
        raise ValueError(f"Unexpected activation: {act}")


def instruction_model_inference(model, input_data):
    import numpy as np

    # Convert weights, bias, and parameters into float32 numpy arrays.
    weights = [np.array(w, dtype=np.float32) for w in model["weights"]]
    bias = [np.array(b, dtype=np.float32) for b in model["bias"]]
    parameters = [np.array(p, dtype=np.float32) for p in model["parameters"]]
    buffer_sizes = model["buffer_sizes"]
    instructions = model["instructions"]
    buffers = [None] * len(buffer_sizes)

    # Initialize the first buffers from input_data (can be a list or a single array).
    if isinstance(input_data, list):
        for i, data in enumerate(input_data):
            buffers[i] = data.copy()
            if buffers[i].shape[1] != buffer_sizes[i]:
                raise ValueError(
                    f"Input data shape does not match the expected shape. Expected {buffer_sizes[i]} but got {buffers[i].shape[1]}"
                )
    else:
        # When input_data is a single numpy array, we need to partition it into the correct initial buffers.
        total_features = input_data.shape[1]
        accumulated = 0
        accumulated_capacities = []
        num_buffers = 0
        valid = False

        # Validate that the input feature size can be exactly partitioned into one or more buffers.
        for size in buffer_sizes:
            accumulated += size
            accumulated_capacities.append(accumulated)
            num_buffers += 1
            if accumulated == total_features:
                valid = True
                break
            elif accumulated > total_features:
                break

        if not valid:
            raise ValueError(
                f"Invalid input feature size: expected one of the cumulative capacities {accumulated_capacities} but got {total_features}"
            )

        # Partition the input_data into the first 'num_buffers' buffers.
        start = 0
        for i in range(num_buffers):
            end = start + buffer_sizes[i]
            buffers[i] = input_data[:, start:end].copy()
            start = end

    # Process each instruction.
    for instruction in instructions:
        if instruction["type"] == "ACTIVATION":
            output_index = instruction["input"]
            apply_activation(instruction, buffers, output_index)
            continue
        # --- New instructions that expect a list of input indices ---
        elif instruction["type"] == "ADD_ELEMENTWISE_BUFFERS":
            # "input" is a list of buffer indices
            input_indices = instruction["input"]
            # Get each buffer and stack along a new axis (assumes all buffers have equal shape)
            stacked = np.stack([buffers[i] for i in input_indices], axis=0)
            result = np.sum(stacked, axis=0)
            output_index = instruction["output"]
            # Compute elementwise addition across the stacked buffers    # This ensures "deploy" only runs on valid version tags in the form release-x.x.x
            buffers[output_index] = np.array(result, dtype=np.float32)
            continue

        elif instruction["type"] == "MULTIPLY_ELEMENTWISE_BUFFERS":
            # "input" is a list of buffer indices
            input_indices = instruction["input"]
            stacked = np.stack([buffers[i] for i in input_indices], axis=0)
            # Compute elementwise multiplication across the stacked buffers.
            result = np.prod(stacked, axis=0)
            output_index = instruction["output"]
            buffers[output_index] = np.array(result, dtype=np.float32)
            continue

        # --- Existing instructions (where "input" is a single index) --
        # In these cases we assume "input" is an integer.
        input_index = instruction["input"]
        # Get the buffer corresponding to the single input for this instruction.
        input_data_buffer = buffers[input_index]

        if instruction["type"] == "COPY_MASKED":
            output_index = instruction["output"]
            indexes = instruction["indexes"]
            if buffers[output_index] is None:
                buffers[output_index] = np.zeros(
                    (input_data_buffer.shape[0], buffer_sizes[output_index]),
                    dtype=np.float32,
                )
            # Copy only the selected indices.
            buffers[output_index][:, list(range(len(indexes)))] = input_data_buffer[
                :, indexes
            ]

        elif instruction["type"] == "COPY":
            output_index = instruction["output"]
            internal_index = instruction["internal_index"]
            size = buffer_sizes[input_index]
            if buffers[output_index] is None:
                buffers[output_index] = np.zeros(
                    (input_data_buffer.shape[0], buffer_sizes[output_index]),
                    dtype=np.float32,
                )
            buffers[output_index][:, internal_index : internal_index + size] = (
                input_data_buffer[:, :]
            )

        elif instruction["type"] == "MAP_TRANSFORM":
            output_index = instruction["output"]
            internal_input_index = instruction["internal_input_index"]
            internal_output_index = instruction["internal_output_index"]
            map_index = instruction["map"]
            default = instruction["default"]
            size = instruction["size"]
            map_data = {int(k): v for k, v in model["maps"][map_index].items()}

            if buffers[output_index] is None:
                buffers[output_index] = np.zeros(
                    (input_data_buffer.shape[0], buffer_sizes[output_index]),
                    dtype=np.float32,
                )

            buffers[output_index][
                :, internal_output_index : internal_output_index + size
            ] = np.array(
                [
                    np.array(map_data.get(i.item(), default))
                    for i in input_data_buffer[
                        :, internal_input_index : internal_input_index + 1
                    ].flatten()
                ]
            ).reshape(-1, size)

        elif instruction["type"] == "DOT":
            w_index = instruction["weights"]
            output_index = instruction["output"]
            buffers[output_index] = (
                input_data_buffer @ weights[w_index].T + bias[w_index]
            )
            apply_activation(instruction, buffers, output_index)

        elif instruction["type"] == "ATTENTION":
            w_index = instruction["weights"]
            output_index = instruction["output"]
            key_index = instruction["key"]
            key_data = buffers[key_index]
            buffers[output_index] = key_data @ weights[w_index].T + bias[w_index]
            x = buffers[output_index]
            x = x - np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x)
            buffers[output_index] = exp_x / np.sum(exp_x, axis=1, keepdims=True)
            buffers[output_index] = buffers[output_index] * input_data_buffer

        elif instruction["type"] == "ADD_ELEMENTWISE":
            param = parameters[instruction["parameters"]]
            buffers[input_index] = input_data_buffer + param

        elif instruction["type"] == "MUL_ELEMENTWISE":
            param = parameters[instruction["parameters"]]
            buffers[input_index] = input_data_buffer * param

        # Ensure the source buffer (still unassigned to an output) is float32.
        buffers[input_index] = np.array(buffers[input_index], dtype=np.float32)

    return buffers


def validate_instruction_model(model):
    input_data_all = np.array(model["validation_data"]["inputs"])
    output_data_all = np.array(model["validation_data"]["expected_outputs"])
    buffers = instruction_model_inference(model, input_data_all)
    output_inference = buffers[-1]
    if not np.allclose(output_data_all, output_inference, atol=1e-6):
        print("output_data_all shape:", output_data_all.shape)
        print("output_inference shape:", output_inference.shape)
        print("MAX ERROR: ", np.max(np.abs(output_data_all - output_inference)))
        print("MEAN ERROR: ", np.mean(np.abs(output_data_all - output_inference)))
        print("All ERROR: ", np.abs(output_data_all - output_inference))

    assert np.allclose(output_data_all, output_inference, atol=1e-6)


def concatenate_instruction_models(models, start_from_first=False):
    features = models[0]["features"]
    # Start with the first model's buffer_sizes. (Assume at least one model.)
    concatenated_buffer_sizes = models[0]["buffer_sizes"][:]
    for model in models[1:]:
        # Ensure that the last buffer in the previous model matches the first buffer of the next.
        assert concatenated_buffer_sizes[-1] == model["buffer_sizes"][0]
        concatenated_buffer_sizes += model["buffer_sizes"][1:]

    if start_from_first:
        new_model = models[0]
        current_buffer_offset = len(new_model["buffer_sizes"]) - 1
        new_model["buffer_sizes"] = concatenated_buffer_sizes
    else:
        new_model = {
            "features": features,
            "buffer_sizes": concatenated_buffer_sizes,
            "instructions": [],
            "weights": [],
            "bias": [],
            "parameters": [],
            "maps": [],
        }
        current_buffer_offset = 0

    for model in models[int(start_from_first) :]:
        w_offset = len(new_model["weights"])
        p_offset = len(new_model["parameters"])
        m_offset = len(new_model["maps"])
        for instruction in model["instructions"]:
            new_instruction = instruction.copy()
            if "weights" in new_instruction:
                new_instruction["weights"] += w_offset
            if "parameters" in new_instruction:
                new_instruction["parameters"] += p_offset
            if "map" in new_instruction:
                new_instruction["map"] += m_offset
            if "input" in new_instruction:
                if isinstance(new_instruction["input"], list):
                    new_instruction["input"] = [
                        x + current_buffer_offset for x in new_instruction["input"]
                    ]
                else:
                    new_instruction["input"] += current_buffer_offset
            if "output" in new_instruction:
                new_instruction["output"] += current_buffer_offset
            if "key" in new_instruction:
                new_instruction["key"] += current_buffer_offset
            new_model["instructions"].append(new_instruction)

        new_model["weights"] += model["weights"]
        new_model["bias"] += model["bias"]
        new_model["parameters"] += model["parameters"]
        new_model["maps"] += model["maps"]

        current_buffer_offset += len(model["buffer_sizes"]) - 1
    return new_model


def create_instructions_model_from_transformation_list(features, transformation_list):
    new_size = (
        len(features)
        + sum(i["size"] for i in transformation_list if "to" in i)
        - sum(1 for i in transformation_list if "delete" in i)
    )

    transformation_list = sorted(
        transformation_list, key=lambda x: "delete" in x, reverse=True
    )

    instructions = []
    new_features = features.copy()
    keep_indexes = [*range(len(features))]
    for transformation in transformation_list:
        if "delete" in transformation:
            new_features.remove(transformation["delete"])
            keep_indexes.remove(features.index(transformation["delete"]))
        elif "to" in transformation:
            new_features.extend(transformation["to"])

    instructions.append(
        {
            "type": "COPY_MASKED",
            "input": 0,
            "output": 1,
            "indexes": keep_indexes,
        }
    )
    maps = []
    for transformation in transformation_list:
        if "to" in transformation:
            from_idx = features.index(transformation["from"])

            instructions.append(
                {
                    "type": "MAP_TRANSFORM",
                    "input": 0,
                    "output": 1,
                    "internal_input_index": from_idx,
                    "internal_output_index": new_features.index(
                        transformation["to"][0]
                    ),
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
