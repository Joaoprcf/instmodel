from instmodel.model import (
    Attention,
    Concatenate,
    Dense,
    InputBuffer,
    ModelGraph,
    SingleIdEmbeddings,
    MultiIdEmbeddings,
    Add,
    ReduceSum,
    ScaleVectorized,
    ShiftVectorized,
    ff_model,
    create_instruction_model,
    create_model_graph,
    validate_keras_model,
    NOT_INPLACE,
    NO_BATCH_NORM,
)


from instmodel.instruction_model import (
    validate_instruction_model,
    instruction_model_inference,
    generate_features,
)

import json
import numpy as np
import pandas as pd
import tensorflow as tf

# Disable TF32 to ensure full float32 precision on GPU (matches CPU/NumPy precision)
tf.config.experimental.enable_tensor_float_32_execution(False)


def test_simple_dense_model():
    """
    Tests a simple feed-forward model with three Dense layers:
      - The first Dense uses ReLU activation.
      - The second Dense uses ReLU activation.
      - The third Dense uses Sigmoid activation.
    Then validates that the exported instruction model matches the Keras outputs.
    """

    input_buffer = InputBuffer(4)
    hidden = Dense(8, activation="relu", name="hidden_relu_1")(input_buffer)
    hidden = Dense(6, activation="relu", name="hidden_relu_2")(hidden)
    output = Dense(1, activation="sigmoid", name="output_sigmoid")(hidden)

    model_graph = ModelGraph(input_buffer, output)
    model_graph.compile(optimizer="adam", loss="binary_crossentropy")

    # Generate dummy data for demonstration.
    x_data = np.random.random((10, 4))
    y_data = np.random.randint(0, 2, size=(10, 1))  # random 0/1 labels

    # Train for one epoch.
    model_graph.fit(x_data, y_data, epochs=1, verbose=0)

    # Export the trained model to an instruction model.
    instruction_model = model_graph.create_instruction_model()

    # Compare Keras predictions to the instruction-model predictions.
    keras_pred = model_graph.predict(x_data, verbose=0)
    instruction_model["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": keras_pred.tolist(),
    }

    # Validate to ensure both models produce the same output.
    validate_instruction_model(instruction_model)
    validate_keras_model(model_graph.get_keras(), instruction_model["validation_data"])

    print(
        "Simple Dense model validation successful: Instruction model matches Keras output."
    )

    del instruction_model["weights"]
    del instruction_model["bias"]
    del instruction_model["parameters"]
    del instruction_model["maps"]
    del instruction_model["validation_data"]

    # If you want to test the exact instruction set, you can assert something like:
    assert instruction_model["buffer_sizes"] == [4, 8, 6, 1]
    assert instruction_model["instructions"] == [
        {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"},
        {"type": "DOT", "input": 1, "output": 2, "weights": 1, "activation": "RELU"},
        {"type": "DOT", "input": 2, "output": 3, "weights": 2, "activation": "SIGMOID"},
    ]

    # This is the compact format to create a simple feed-forward model.
    model_graph_copy = ff_model([4, 8, 6, 1], NO_BATCH_NORM, "rrs")

    instruction_model_copy = model_graph_copy.create_instruction_model()

    del instruction_model_copy["weights"]
    del instruction_model_copy["bias"]
    del instruction_model_copy["parameters"]
    del instruction_model_copy["maps"]

    assert instruction_model_copy == instruction_model


def test_complex_attention_model():
    """
    Tests a complex attention-based model and validates both the instruction model and Keras model.
    """
    i_target = InputBuffer(20)
    i_key = Dense(10, name="key_dense")(i_target)
    attn = Attention(name="attention")
    attn_out = attn([i_target, i_key])
    d_out = Dense(1, activation="sigmoid", name="dense_output")(attn_out)

    model_graph = create_model_graph(i_target, d_out)

    model_graph.compile(optimizer="adam", loss="mse")
    x_data = np.random.random((50, 20))
    y_data = np.random.random((50, 1))
    model_graph.fit(x_data, y_data, epochs=2, verbose=0)

    result = create_instruction_model(i_target, d_out)

    keras_pred = model_graph.predict(x_data, verbose=0)
    result["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": keras_pred.tolist(),
    }

    validate_instruction_model(result)
    validate_keras_model(model_graph.get_keras(), result["validation_data"])

    print("Validation successful: Instruction model output matches expected output.")

    del result["weights"]
    del result["bias"]
    del result["parameters"]
    del result["maps"]
    del result["validation_data"]

    assert result == {
        "features": [],
        "buffer_sizes": [20, 10, 20, 1],
        "instructions": [
            {"type": "DOT", "input": 0, "output": 1, "weights": 0},
            {"type": "ATTENTION", "input": 0, "key": 1, "output": 2, "weights": 1},
            {
                "type": "DOT",
                "input": 2,
                "output": 3,
                "weights": 2,
                "activation": "SIGMOID",
            },
        ],
    }


def test_nested_model():
    """
    Tests a nested model structure and validates the resulting instruction model.
    """
    model = ff_model([3, 3, 3], NOT_INPLACE, "ll")

    main_input = InputBuffer(3)

    first_iteration = model(main_input)
    second_iteration = model(first_iteration)

    concat = Concatenate()([main_input, first_iteration, second_iteration])

    dense_out = Dense(1)(concat)

    final_model = ModelGraph(main_input, dense_out)

    final_model.compile(optimizer="adam", loss="mse")
    x_data = np.random.random((50, 3)) + 2
    y_data = np.random.random((50, 1))

    final_model.fit(x_data, y_data, epochs=1, verbose=0)
    result = final_model.create_instruction_model()

    y_pred = final_model.predict(x_data)

    result["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": y_pred.tolist(),
    }

    validate_instruction_model(result)
    validate_keras_model(final_model.get_keras(), result["validation_data"])

    del result["weights"]
    del result["bias"]
    del result["parameters"]
    del result["maps"]
    del result["validation_data"]

    assert result == {
        "features": [],
        "buffer_sizes": [3, 3, 3, 3, 3, 3, 3, 9, 1],
        "instructions": [
            {"type": "COPY", "input": 0, "output": 1, "internal_index": 0},
            {"type": "ADD_ELEMENTWISE", "input": 1, "parameters": 0},
            {"type": "MUL_ELEMENTWISE", "input": 1, "parameters": 1},
            {"type": "ADD_ELEMENTWISE", "input": 1, "parameters": 2},
            {"type": "DOT", "input": 1, "output": 2, "weights": 0},
            {"type": "DOT", "input": 2, "output": 3, "weights": 1},
            {"type": "COPY", "input": 3, "output": 4, "internal_index": 0},
            {"type": "ADD_ELEMENTWISE", "input": 4, "parameters": 0},
            {"type": "MUL_ELEMENTWISE", "input": 4, "parameters": 1},
            {"type": "ADD_ELEMENTWISE", "input": 4, "parameters": 2},
            {"type": "DOT", "input": 4, "output": 5, "weights": 0},
            {"type": "DOT", "input": 5, "output": 6, "weights": 1},
            {"type": "COPY", "input": 0, "output": 7, "internal_index": 0},
            {"type": "COPY", "input": 3, "output": 7, "internal_index": 3},
            {"type": "COPY", "input": 6, "output": 7, "internal_index": 6},
            {"type": "DOT", "input": 7, "output": 8, "weights": 2},
        ],
    }


def test_ff_models():
    """
    Tests composite feed-forward models with concatenation and validates the instruction model.
    """
    input_buffer = InputBuffer(3)

    path1 = ff_model([3, 5, 3], NOT_INPLACE, "rs")(input_buffer)

    path2 = ff_model([3, 4, 6], NOT_INPLACE, "tr")(input_buffer)

    concat = Concatenate()([path1, path2])

    out = ff_model([concat.os, 2, 1], NO_BATCH_NORM, "ts")(concat)

    final_model = ModelGraph(input_buffer, out)

    result = final_model.create_instruction_model()

    del result["weights"]
    del result["bias"]
    del result["parameters"]
    del result["maps"]

    assert result == {
        "features": [],
        "buffer_sizes": [3, 3, 5, 3, 3, 4, 6, 9, 2, 1],
        "instructions": [
            {"type": "COPY", "input": 0, "output": 1, "internal_index": 0},
            {"type": "ADD_ELEMENTWISE", "input": 1, "parameters": 0},
            {"type": "MUL_ELEMENTWISE", "input": 1, "parameters": 1},
            {"type": "ADD_ELEMENTWISE", "input": 1, "parameters": 2},
            {
                "type": "DOT",
                "input": 1,
                "output": 2,
                "weights": 0,
                "activation": "RELU",
            },
            {
                "type": "DOT",
                "input": 2,
                "output": 3,
                "weights": 1,
                "activation": "SIGMOID",
            },
            {"type": "COPY", "input": 0, "output": 4, "internal_index": 0},
            {"type": "ADD_ELEMENTWISE", "input": 4, "parameters": 3},
            {"type": "MUL_ELEMENTWISE", "input": 4, "parameters": 4},
            {"type": "ADD_ELEMENTWISE", "input": 4, "parameters": 5},
            {
                "type": "DOT",
                "input": 4,
                "output": 5,
                "weights": 2,
                "activation": "TANH",
            },
            {
                "type": "DOT",
                "input": 5,
                "output": 6,
                "weights": 3,
                "activation": "RELU",
            },
            {"type": "COPY", "input": 3, "output": 7, "internal_index": 0},
            {"type": "COPY", "input": 6, "output": 7, "internal_index": 3},
            {
                "type": "DOT",
                "input": 7,
                "output": 8,
                "weights": 4,
                "activation": "TANH",
            },
            {
                "type": "DOT",
                "input": 8,
                "output": 9,
                "weights": 5,
                "activation": "SIGMOID",
            },
        ],
    }


def test_slices():
    """
    Tests slicing of an InputBuffer and the concatenation of sliced outputs.
    """
    input_buffer = InputBuffer(3)

    slice1 = input_buffer[1:]
    slice2 = input_buffer[:1]

    concat = Concatenate()([slice1, slice2])

    model = ModelGraph(input_buffer, concat)

    result = model.create_instruction_model()

    del result["weights"]
    del result["bias"]
    del result["parameters"]
    del result["maps"]

    assert result == {
        "features": [],
        "buffer_sizes": [3, 2, 1, 3],
        "instructions": [
            {"type": "COPY_MASKED", "input": 0, "output": 1, "indexes": [1, 2]},
            {"type": "COPY_MASKED", "input": 0, "output": 2, "indexes": [0]},
            {"type": "COPY", "input": 1, "output": 3, "internal_index": 0},
            {"type": "COPY", "input": 2, "output": 3, "internal_index": 2},
        ],
    }


def test_feature_computing():
    """
    Tests the feature computing functionality of the model.
    """
    input_buffers = [InputBuffer(1), InputBuffer(1)]

    output = Add()(input_buffers)

    model = ModelGraph(input_buffers, output)

    inst_model = model.create_instruction_model(["feature1", "feature2"])

    with open("tests/files/instmodel.json", "r") as f:
        file_content = json.load(f)

    assert file_content == inst_model

    result = instruction_model_inference(inst_model, [np.array([[1]]), np.array([[2]])])

    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

    # Create a DataFrame with the specified dataset.
    simple_dataset = pd.DataFrame({"feature1": [1, 0.5], "feature2": [2, -5.0]})

    simple_dataset = generate_features(
        "tests/files/instmodel.json", simple_dataset, ["feature3"]
    )

    assert simple_dataset.columns.tolist() == ["feature1", "feature2", "feature3"]

    assert simple_dataset["feature3"].tolist() == [3, -4.5]


def test_embeddings(capsys):
    """
    Tests the embedding functionality of the model.
    """

    # Generate dummy data
    x_data = np.random.random((50, 3))
    # Ensure the second column contains only integers 1 or 2
    x_data[:, 1] = np.random.randint(1, 4, size=(50,))
    # Calculate y_data based on x_data conditions
    conditions = [x_data[:, 1] == 1, x_data[:, 1] == 2]
    choices = [
        x_data[:, 0] + x_data[:, 2],  # Choice for condition 1
        x_data[:, 0] - x_data[:, 2] * 0.5,  # Choice for condition 2
    ]
    default_choice = -x_data[:, 0] + x_data[:, 2] * 1.5  # Default choice

    y_data = np.select(conditions, choices, default=default_choice).reshape(-1, 1)

    features = ["non-id1", "id1", "non-id2"]
    input_buffer = InputBuffer(3)
    non_id_features = input_buffer[0, 2]
    id_features = input_buffer[(1,)]

    converted_to_embedding = SingleIdEmbeddings([1, 2], 3)(id_features)

    concatenated = Concatenate()([non_id_features, converted_to_embedding])

    dense_mid = Dense(8, activation="relu")(concatenated)

    dense_out = Dense(1)(dense_mid)

    model = ModelGraph(input_buffer, dense_out)

    model.compile(optimizer="adam", loss="mse")

    model.fit(x_data, y_data, epochs=20, verbose=0)

    # Predict after training
    pred_after = model.predict(x_data, verbose=0)

    instmodel = model.create_instruction_model(features)

    *_, inst_output_y = instruction_model_inference(instmodel, x_data)

    # Check if the instruction model output is close to the Keras model output
    assert np.allclose(inst_output_y, pred_after, atol=1e-6), (
        "Instruction model output does not match Keras output after training."
    )

    instmodel["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": pred_after.tolist(),
    }

    validate_instruction_model(instmodel)

    assert len(instmodel["weights"]) == 2
    del instmodel["weights"]
    assert len(instmodel["bias"]) == 2
    del instmodel["bias"]
    assert len(instmodel["maps"]) == 1
    assert instmodel["maps"][0].keys() == {1, 2}
    del instmodel["maps"]
    del instmodel["parameters"]
    del instmodel["validation_data"]
    assert len(instmodel["instructions"]) == 7
    assert len(instmodel["instructions"][2]["default"]) == 3
    del instmodel["instructions"][2]["default"]

    assert instmodel == {
        "features": features,
        "buffer_sizes": [
            len(features),
            2,
            1,
            converted_to_embedding.os,
            len(features) + converted_to_embedding.os - 1,
            8,
            1,
        ],
        "instructions": [
            {"type": "COPY_MASKED", "input": 0, "output": 1, "indexes": [0, 2]},
            {"type": "COPY_MASKED", "input": 0, "output": 2, "indexes": [1]},
            {
                "type": "MAP_TRANSFORM",
                "input": 2,
                "output": 3,
                "internal_input_index": 0,
                "internal_output_index": 0,
                "map": 0,
                "size": 3,
            },
            {"type": "COPY", "input": 1, "output": 4, "internal_index": 0},
            {"type": "COPY", "input": 3, "output": 4, "internal_index": 2},
            {
                "type": "DOT",
                "input": 4,
                "output": 5,
                "weights": 0,
                "activation": "RELU",
            },
            {"type": "DOT", "input": 5, "output": 6, "weights": 1},
        ],
    }

    encoded_input = MultiIdEmbeddings([1], [[1, 2]], [4])(input_buffer)

    dense_mid = Dense(8, activation="relu")(encoded_input)

    dense_out = Dense(1)(dense_mid)

    encoded_model = ModelGraph(input_buffer, dense_out)

    encoded_model.compile(optimizer="adam", loss="mse")

    encoded_model.fit(x_data, y_data, epochs=20, verbose=0)

    # Predict after training
    pred_after = encoded_model.predict(x_data, verbose=0)

    complex_instmodel = encoded_model.create_instruction_model(features)

    *_, inst_output_y = instruction_model_inference(complex_instmodel, x_data)

    assert np.allclose(inst_output_y, pred_after, atol=1e-6), (
        "Instruction model output does not match Keras output after training."
    )

    complex_instmodel["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": pred_after.tolist(),
    }

    validate_instruction_model(complex_instmodel)

    assert len(complex_instmodel["weights"]) == 2
    del complex_instmodel["weights"]
    assert len(complex_instmodel["bias"]) == 2
    del complex_instmodel["bias"]
    assert len(complex_instmodel["maps"]) == 1
    assert complex_instmodel["maps"][0].keys() == {1, 2}
    del complex_instmodel["maps"]
    del complex_instmodel["parameters"]
    del complex_instmodel["validation_data"]
    assert len(complex_instmodel["instructions"]) == 4
    assert len(complex_instmodel["instructions"][1]["default"]) == 4
    del complex_instmodel["instructions"][1]["default"]

    assert complex_instmodel == {
        "features": features,
        "buffer_sizes": [len(features), 6, 8, 1],
        "instructions": [
            {"type": "COPY_MASKED", "input": 0, "output": 1, "indexes": [0, 2]},
            {
                "type": "MAP_TRANSFORM",
                "input": 0,
                "output": 1,
                "internal_input_index": 1,
                "internal_output_index": 2,
                "map": 0,
                "size": 4,
            },
            {
                "type": "DOT",
                "input": 1,
                "output": 2,
                "weights": 0,
                "activation": "RELU",
            },
            {"type": "DOT", "input": 2, "output": 3, "weights": 1},
        ],
    }


def test_scale_and_shift_vectorized():
    """
    Tests ScaleVectorized and ShiftVectorized operations with in_place=True.
    """
    input_buffer = InputBuffer(3)

    # Apply scaling by [2, 0.5, 3] then shift by [1, -1, 0] (in-place)
    scaled = ScaleVectorized([2.0, 0.5, 3.0], in_place=True)(input_buffer)
    shifted = ShiftVectorized([1.0, -1.0, 0.0], in_place=True)(scaled)

    model = ModelGraph(input_buffer, shifted)

    result = model.create_instruction_model()

    # Check the instruction structure
    del result["weights"]
    del result["bias"]
    del result["maps"]

    # Parameters should contain the scale and shift vectors
    assert len(result["parameters"]) == 2
    assert result["parameters"][0] == [2.0, 0.5, 3.0]
    assert result["parameters"][1] == [1.0, -1.0, 0.0]

    del result["parameters"]

    # Both operations are in-place on buffer 0
    assert result == {
        "features": [],
        "buffer_sizes": [3],
        "instructions": [
            {"type": "MUL_ELEMENTWISE", "input": 0, "parameters": 0},
            {"type": "ADD_ELEMENTWISE", "input": 0, "parameters": 1},
        ],
    }

    # Test numerical correctness
    x_data = np.array([[1.0, 2.0, 3.0], [0.0, -4.0, 2.0]])
    keras_pred = model.predict(x_data, verbose=0)

    # Expected: (x * scale) + shift
    expected = x_data * np.array([2.0, 0.5, 3.0]) + np.array([1.0, -1.0, 0.0])
    assert np.allclose(keras_pred, expected, atol=1e-6)

    # Validate instruction model matches Keras
    inst_model = model.create_instruction_model()
    inst_model["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": keras_pred.tolist(),
    }
    validate_instruction_model(inst_model)


def test_scale_and_shift_not_inplace():
    """
    Tests ScaleVectorized and ShiftVectorized with in_place=False.
    """
    input_buffer = InputBuffer(2)

    scaled = ScaleVectorized([2.0, 3.0], in_place=False)(input_buffer)
    shifted = ShiftVectorized([10.0, 20.0], in_place=False)(scaled)

    model = ModelGraph(input_buffer, shifted)

    result = model.create_instruction_model()

    del result["weights"]
    del result["bias"]
    del result["maps"]
    del result["parameters"]

    # Not in-place means new buffers are allocated
    assert result == {
        "features": [],
        "buffer_sizes": [2, 2, 2],
        "instructions": [
            {"type": "COPY", "input": 0, "output": 1, "internal_index": 0},
            {"type": "MUL_ELEMENTWISE", "input": 1, "parameters": 0},
            {"type": "COPY", "input": 1, "output": 2, "internal_index": 0},
            {"type": "ADD_ELEMENTWISE", "input": 2, "parameters": 1},
        ],
    }


def test_scale_vectorized_scalar():
    """
    Tests ScaleVectorized with a single scalar value that broadcasts to all elements.
    """
    input_buffer = InputBuffer(4)

    # Scalar 2.0 should multiply all 4 elements
    scaled = ScaleVectorized(2.0, in_place=True)(input_buffer)
    model = ModelGraph(input_buffer, scaled)

    result = model.create_instruction_model()

    # Parameter should be expanded to [2.0, 2.0, 2.0, 2.0]
    assert result["parameters"][0] == [2.0, 2.0, 2.0, 2.0]

    # Numerical test
    x_data = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 0.0, 2.0]])
    keras_pred = model.predict(x_data, verbose=0)
    expected = x_data * 2.0
    assert np.allclose(keras_pred, expected, atol=1e-6)

    # Validate instruction model
    inst_model = model.create_instruction_model()
    inst_model["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": keras_pred.tolist(),
    }
    validate_instruction_model(inst_model)


def test_shift_vectorized_scalar():
    """
    Tests ShiftVectorized with a single scalar value that broadcasts to all elements.
    """
    input_buffer = InputBuffer(3)

    # Scalar 5.0 should be added to all 3 elements
    shifted = ShiftVectorized(5.0, in_place=True)(input_buffer)
    model = ModelGraph(input_buffer, shifted)

    result = model.create_instruction_model()

    # Parameter should be expanded to [5.0, 5.0, 5.0]
    assert result["parameters"][0] == [5.0, 5.0, 5.0]

    # Numerical test
    x_data = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 10.0]])
    keras_pred = model.predict(x_data, verbose=0)
    expected = x_data + 5.0
    assert np.allclose(keras_pred, expected, atol=1e-6)

    # Validate instruction model
    inst_model = model.create_instruction_model()
    inst_model["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": keras_pred.tolist(),
    }
    validate_instruction_model(inst_model)


def test_reduce_sum():
    """
    Tests ReduceSum layer that sums all elements of the input buffer.
    Input: (batch, N) → Output: (batch, 1)
    """
    input_buffer = InputBuffer(5)
    summed = ReduceSum()(input_buffer)

    model = ModelGraph(input_buffer, summed)

    # Test with sample data
    x_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 1.5, 2.5, 3.5, 4.5]], dtype=np.float32)

    # Expected: sum along last axis with keepdims=True
    expected = np.array([[15.0], [12.5]], dtype=np.float32)

    keras_pred = model.predict_on_batch(x_data)
    assert np.allclose(keras_pred, expected, atol=1e-6)

    # Validate instruction model matches Keras
    inst_model = model.create_instruction_model()

    # Verify instruction structure
    assert inst_model["buffer_sizes"] == [5, 1]
    assert inst_model["instructions"] == [
        {"type": "REDUCE_SUM", "input": 0, "output": 1}
    ]

    inst_model["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": keras_pred.tolist(),
    }
    validate_instruction_model(inst_model)


def test_reduce_sum_in_pipeline():
    """
    Tests ReduceSum as part of a larger computation pipeline.
    """
    input_buffer = InputBuffer(4)
    hidden = Dense(8, activation="relu")(input_buffer)
    summed = ReduceSum()(hidden)

    model = ModelGraph(input_buffer, summed)

    x_data = np.random.randn(10, 4).astype(np.float32)
    keras_pred = model.predict_on_batch(x_data)

    # Output should be (10, 1)
    assert keras_pred.shape == (10, 1)

    # Validate instruction model matches Keras
    inst_model = model.create_instruction_model()
    inst_model["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": keras_pred.tolist(),
    }
    validate_instruction_model(inst_model)


def test_gelu_activation():
    """
    Tests the GeLU activation function matching TensorFlow's default (exact erf-based).
    Verifies both the Dense layer with gelu and the ff_model shorthand 'g'.
    """
    from scipy.special import erf

    # Test 1: Simple model with GeLU activation
    input_buffer = InputBuffer(4)
    hidden = Dense(8, activation="gelu", name="hidden_gelu")(input_buffer)
    output = Dense(1, activation="sigmoid", name="output_sigmoid")(hidden)

    model_graph = ModelGraph(input_buffer, output)
    model_graph.compile(optimizer="adam", loss="binary_crossentropy")

    x_data = np.random.random((10, 4)).astype(np.float32)
    y_data = np.random.randint(0, 2, size=(10, 1))

    model_graph.fit(x_data, y_data, epochs=1, verbose=0)

    instruction_model = model_graph.create_instruction_model()

    keras_pred = model_graph.predict(x_data, verbose=0)
    instruction_model["validation_data"] = {
        "inputs": x_data.tolist(),
        "expected_outputs": keras_pred.tolist(),
    }

    validate_instruction_model(instruction_model)
    validate_keras_model(model_graph.get_keras(), instruction_model["validation_data"])

    # Check instruction structure
    assert instruction_model["instructions"][0]["activation"] == "GELU"
    assert instruction_model["instructions"][1]["activation"] == "SIGMOID"

    # Test 2: ff_model with 'g' shorthand for gelu
    model_graph_ff = ff_model([4, 8, 1], NO_BATCH_NORM, "gs")

    instruction_model_ff = model_graph_ff.create_instruction_model()

    # Verify instruction structure matches
    assert instruction_model_ff["instructions"][0]["activation"] == "GELU"
    assert instruction_model_ff["instructions"][1]["activation"] == "SIGMOID"

    # Test 3: Verify NumPy GeLU matches TensorFlow exactly
    test_input = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

    # TensorFlow GeLU (default, exact)
    tf_gelu = tf.nn.gelu(test_input, approximate=False).numpy()

    # NumPy GeLU using our implementation
    numpy_gelu = test_input * 0.5 * (1 + erf(test_input / np.sqrt(2)))

    assert np.allclose(tf_gelu, numpy_gelu, atol=1e-6), (
        f"NumPy GeLU does not match TensorFlow GeLU. "
        f"TF: {tf_gelu}, NumPy: {numpy_gelu}"
    )

    # Test 4: Verify instruction model inference matches Keras with GeLU
    input_buffer2 = InputBuffer(5)
    gelu_layer = Dense(10, activation="gelu")(input_buffer2)
    output2 = Dense(3)(gelu_layer)

    model2 = ModelGraph(input_buffer2, output2)

    x_data2 = np.random.randn(20, 5).astype(np.float32)
    keras_pred2 = model2.predict(x_data2, verbose=0)

    inst_model2 = model2.create_instruction_model()
    *_, inst_output = instruction_model_inference(inst_model2, x_data2)

    assert np.allclose(inst_output, keras_pred2, atol=1e-6), (
        "Instruction model GeLU output does not match Keras output."
    )
