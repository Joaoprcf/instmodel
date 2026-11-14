import copy

import numpy as np
import pandas as pd
import pytest

from instmodel.instruction_model import (
    concatenate_instruction_models,
    create_instructions_model_from_transformation_list,
    generate_features,
    instruction_model_inference,
    validate_instruction_model,
)

DENSE = {
    "features": ["a", "b"],
    "buffer_sizes": [2, 2, 1],
    "instructions": [
        {"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "RELU"},
        {"type": "COPY_MASKED", "input": 1, "output": 2, "indexes": [1]},
    ],
    "weights": [[[1.0, 0.0], [0.0, 2.0]]],
    "bias": [[0.0, 1.0]],
    "parameters": [],
    "maps": [],
}


def test_generate_features():
    df = pd.DataFrame({"feature1": [1, 0.5], "feature2": [2, -5.0]})
    out = generate_features("tests/files/instmodel.json", df, ["feature3"])
    assert out.columns.tolist() == ["feature1", "feature2", "feature3"]
    assert out["feature3"].tolist() == [3, -4.5]
    with pytest.raises(ValueError, match="Model not found"):
        generate_features("tests/files/missing.json", df, ["feature3"])


def test_input_partitioning():
    x = np.array([[1.0, -3.0]])
    assert instruction_model_inference(DENSE, x)[-1].tolist() == [[0.0]]
    with pytest.raises(ValueError, match="cumulative capacities"):
        instruction_model_inference(DENSE, np.ones((1, 3)))
    with pytest.raises(ValueError, match="Expected 2 but got 3"):
        instruction_model_inference(DENSE, [np.ones((1, 3))])


def test_validate_instruction_model_raises_on_mismatch():
    model = dict(DENSE, validation_data={"inputs": [[1.0, 2.0]], "expected_outputs": [[5.0]]})
    validate_instruction_model(model)
    model["validation_data"]["expected_outputs"] = [[4.0]]
    with pytest.raises(AssertionError):
        validate_instruction_model(model)


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"input": -1}, "buffer index -1"),
        ({"output": 3}, "buffer index 3"),
        ({"input": "0"}, "must be an integer"),
        ({"weights": 1}, "weights index 1"),
        ({"type": "BOGUS"}, "Unknown instruction type"),
        ({"activation": "NONE"}, "Unexpected activation"),
    ],
)
def test_malformed_dot_is_rejected(patch, message):
    model = copy.deepcopy(DENSE)
    model["instructions"][0].update(patch)
    with pytest.raises(ValueError, match=message):
        instruction_model_inference(model, np.ones((1, 2)))


@pytest.mark.parametrize(
    "instr, message",
    [
        ({"type": "COPY_MASKED", "input": 0, "output": 2, "indexes": [-1]}, "COPY_MASKED column index -1"),
        ({"type": "COPY_MASKED", "input": 0, "output": 2, "indexes": [0, 1]}, "more columns"),
        ({"type": "COPY", "input": 2, "output": 1, "internal_index": -1}, "COPY offset index -1"),
        ({"type": "COPY", "input": 1, "output": 2, "internal_index": 0}, "past buffer size"),
        ({"type": "ADD_ELEMENTWISE", "input": 0, "parameters": 0}, "parameters index 0"),
        ({"type": "ADD_ELEMENTWISE_BUFFERS", "input": [0, 7], "output": 1}, "buffer index 7"),
        (
            {"type": "MAP_TRANSFORM", "input": 0, "output": 2, "internal_input_index": -2,
             "internal_output_index": 0, "map": 0, "size": 1, "default": [0]},
            "MAP_TRANSFORM column index -2",
        ),
        (
            {"type": "MAP_TRANSFORM", "input": 0, "output": 2, "internal_input_index": 0,
             "internal_output_index": 0, "map": 1, "size": 1, "default": [0]},
            "map index 1",
        ),
    ],
)
def test_out_of_range_references_are_rejected(instr, message):
    model = copy.deepcopy(DENSE)
    model["maps"] = [{"1": [5.0]}]
    model["instructions"] = [instr]
    with pytest.raises(ValueError, match=message):
        instruction_model_inference(model, np.ones((1, 2)))


def test_transformation_list_and_concatenation():
    features = ["a", "id", "b"]
    pre, new_features = create_instructions_model_from_transformation_list(
        features,
        [{"delete": "id"}, {"from": "id", "to": ["e0", "e1"], "map": {"7": [1.0, 2.0]}, "size": 2, "default": [0.0, -1.0]}],
    )
    assert new_features == ["a", "b", "e0", "e1"]
    x = np.array([[1.0, 7.0, 2.0], [3.0, 9.0, 4.0]])
    assert instruction_model_inference(pre, x)[-1].tolist() == [[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 0.0, -1.0]]

    head = {
        "features": [],
        "buffer_sizes": [4, 1],
        "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0}],
        "weights": [[[1.0, 1.0, 1.0, 1.0]]],
        "bias": [[0.5]],
        "parameters": [],
        "maps": [],
    }
    chained = concatenate_instruction_models([pre, head])
    assert chained["buffer_sizes"] == [3, 4, 1]
    assert chained["instructions"][-1] == {"type": "DOT", "input": 1, "output": 2, "weights": 0}
    assert instruction_model_inference(chained, x)[-1].tolist() == [[6.5], [6.5]]
    assert concatenate_instruction_models([copy.deepcopy(pre), head], start_from_first=True) == chained


def test_convert_to_serializable():
    from decimal import Decimal

    from instmodel.training_utils import convert_to_serializable

    out = convert_to_serializable({"w": np.array([[1.5, 2.0]], dtype=np.float32), "i": [np.int64(3)], "s": "x"})
    assert out == {"w": [[1.5, 2.0]], "i": [3], "s": "x"}
    assert isinstance(convert_to_serializable(np.float32(0.1)), Decimal)
