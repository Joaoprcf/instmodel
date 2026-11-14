import importlib
import json

import numpy as np
import pytest

from instmodel.instruction_model import instruction_model_inference, validate_instruction_model


@pytest.fixture(params=["tf", "torch"], scope="module")
def B(request):
    if request.param == "tf":
        tf = pytest.importorskip("tensorflow")
        tf.config.experimental.enable_tensor_float_32_execution(False)
    else:
        pytest.importorskip("torch")
    return importlib.import_module(f"instmodel.{request.param}")


def structure(im):
    return {k: v for k, v in im.items() if k not in ("weights", "bias", "parameters", "maps", "validation_data")}


def roundtrip(model, x, features=None):
    """Exports `model`, checks NumPy inference against the backend on `x`, returns (instruction model, prediction)."""
    pred = model.predict(x, verbose=0)
    im = model.create_instruction_model(features)
    im["validation_data"] = {"inputs": np.asarray(x).tolist(), "expected_outputs": pred.tolist()}
    validate_instruction_model(im)
    return im, pred


def train(model, x, y, loss="mse", epochs=1):
    model.compile(optimizer="adam", loss=loss)
    model.fit(x, y, epochs=epochs, verbose=0)


def dot(i, o, w, act=None):
    d = {"type": "DOT", "input": i, "output": o, "weights": w}
    if act:
        d["activation"] = act
    return d


def copy(i, o, at):
    return {"type": "COPY", "input": i, "output": o, "internal_index": at}


def bn(buf, p):
    return [
        {"type": "ADD_ELEMENTWISE", "input": buf, "parameters": p},
        {"type": "MUL_ELEMENTWISE", "input": buf, "parameters": p + 1},
        {"type": "ADD_ELEMENTWISE", "input": buf, "parameters": p + 2},
    ]


def test_simple_dense_model(B):
    inp = B.InputBuffer(4)
    h = B.Dense(8, activation="relu", name="hidden_relu_1")(inp)
    h = B.Dense(6, activation="relu", name="hidden_relu_2")(h)
    model = B.ModelGraph(inp, B.Dense(1, activation="sigmoid", name="output_sigmoid")(h))
    x = np.random.random((10, 4))
    train(model, x, np.random.randint(0, 2, size=(10, 1)), loss="binary_crossentropy")

    im, _ = roundtrip(model, x)
    B.validate_model(model, im["validation_data"])
    assert structure(im) == {
        "features": [],
        "buffer_sizes": [4, 8, 6, 1],
        "instructions": [dot(0, 1, 0, "RELU"), dot(1, 2, 1, "RELU"), dot(2, 3, 2, "SIGMOID")],
    }
    assert structure(B.ff_model([4, 8, 6, 1], B.NO_BATCH_NORM, "rrs").create_instruction_model()) == structure(im)

    im["validation_data"]["expected_outputs"][0][0] += 1.0
    with pytest.raises(AssertionError):
        B.validate_model(model, im["validation_data"])


def test_complex_attention_model(B):
    target = B.InputBuffer(20)
    key = B.Dense(10, name="key_dense")(target)
    out = B.Dense(1, activation="sigmoid", name="dense_output")(B.Attention(name="attention")([target, key]))
    model = B.create_model_graph(target, out)
    x = np.random.random((50, 20))
    train(model, x, np.random.random((50, 1)), epochs=2)

    result = B.create_instruction_model(target, out)
    pred = model.predict(x, verbose=0)
    result["validation_data"] = {"inputs": x.tolist(), "expected_outputs": pred.tolist()}
    validate_instruction_model(result)
    assert structure(result) == {
        "features": [],
        "buffer_sizes": [20, 10, 20, 1],
        "instructions": [
            dot(0, 1, 0),
            {"type": "ATTENTION", "input": 0, "key": 1, "output": 2, "weights": 1},
            dot(2, 3, 2, "SIGMOID"),
        ],
    }


def test_nested_model(B):
    inner = B.ff_model([3, 3, 3], B.NOT_INPLACE, "ll")
    inp = B.InputBuffer(3)
    first = inner(inp)
    second = inner(first)
    model = B.ModelGraph(inp, B.Dense(1)(B.Concatenate()([inp, first, second])))
    x = np.random.random((50, 3)) + 2
    train(model, x, np.random.random((50, 1)))

    im, _ = roundtrip(model, x)
    assert structure(im) == {
        "features": [],
        "buffer_sizes": [3, 3, 3, 3, 3, 3, 3, 9, 1],
        "instructions": [
            copy(0, 1, 0), *bn(1, 0), dot(1, 2, 0), dot(2, 3, 1),
            copy(3, 4, 0), *bn(4, 0), dot(4, 5, 0), dot(5, 6, 1),
            copy(0, 7, 0), copy(3, 7, 3), copy(6, 7, 6), dot(7, 8, 2),
        ],
    }


def test_ff_models(B):
    inp = B.InputBuffer(3)
    path1 = B.ff_model([3, 5, 3], B.NOT_INPLACE, "rs")(inp)
    path2 = B.ff_model([3, 4, 6], B.NOT_INPLACE, "tr")(inp)
    concat = B.Concatenate()([path1, path2])
    out = B.ff_model([concat.os, 2, 1], B.NO_BATCH_NORM, "ts")(concat)

    assert structure(B.ModelGraph(inp, out).create_instruction_model()) == {
        "features": [],
        "buffer_sizes": [3, 3, 5, 3, 3, 4, 6, 9, 2, 1],
        "instructions": [
            copy(0, 1, 0), *bn(1, 0), dot(1, 2, 0, "RELU"), dot(2, 3, 1, "SIGMOID"),
            copy(0, 4, 0), *bn(4, 3), dot(4, 5, 2, "TANH"), dot(5, 6, 3, "RELU"),
            copy(3, 7, 0), copy(6, 7, 3), dot(7, 8, 4, "TANH"), dot(8, 9, 5, "SIGMOID"),
        ],
    }


def test_ff_model_validates_activation_count(B):
    with pytest.raises(ValueError, match="number of activations"):
        B.ff_model([3, 4, 1], B.NO_BATCH_NORM, "r")


@pytest.mark.parametrize("use_batch_norm", [False, True])
def test_stair_structure(B, use_batch_norm):
    model, layers = B.create_stair_structure(4, [3, 2], use_batch_norm=use_batch_norm)
    assert len(layers) == 2 + use_batch_norm
    im, _ = roundtrip(model, np.random.randn(5, 4).astype(np.float32))
    assert im["buffer_sizes"][-1] == 4 + 3 + 2


def test_slices(B):
    inp = B.InputBuffer(3)
    model = B.ModelGraph(inp, B.Concatenate()([inp[1:], inp[:1]]))
    im, _ = roundtrip(model, np.random.randn(4, 3).astype(np.float32))
    assert structure(im) == {
        "features": [],
        "buffer_sizes": [3, 2, 1, 3],
        "instructions": [
            {"type": "COPY_MASKED", "input": 0, "output": 1, "indexes": [1, 2]},
            {"type": "COPY_MASKED", "input": 0, "output": 2, "indexes": [0]},
            copy(1, 3, 0),
            copy(2, 3, 2),
        ],
    }
    with pytest.raises(TypeError):
        inp["a"]


def test_feature_computing(B):
    inputs = [B.InputBuffer(1), B.InputBuffer(1)]
    im = B.ModelGraph(inputs, B.Add()(inputs)).create_instruction_model(["feature1", "feature2"])
    with open("tests/files/instmodel.json") as f:
        assert json.load(f) == im
    assert [b.item() for b in instruction_model_inference(im, [np.array([[1]]), np.array([[2]])])] == [1, 2, 3]


def test_embeddings(B):
    x = np.random.random((50, 3))
    x[:, 1] = np.random.randint(1, 4, size=(50,))
    y = np.select(
        [x[:, 1] == 1, x[:, 1] == 2],
        [x[:, 0] + x[:, 2], x[:, 0] - x[:, 2] * 0.5],
        default=-x[:, 0] + x[:, 2] * 1.5,
    ).reshape(-1, 1)
    features = ["non-id1", "id1", "non-id2"]
    inp = B.InputBuffer(3)

    embedded = B.SingleIdEmbeddings([1, 2], 3)(inp[(1,)])
    concat = B.Concatenate()([inp[0, 2], embedded])
    model = B.ModelGraph(inp, B.Dense(1)(B.Dense(8, activation="relu")(concat)))
    train(model, x, y, epochs=20)
    im, _ = roundtrip(model, x, features)
    assert (len(im["weights"]), len(im["bias"]), len(im["maps"])) == (2, 2, 1)
    assert im["maps"][0].keys() == {1, 2}
    assert len(im["instructions"][2].pop("default")) == 3
    assert structure(im) == {
        "features": features,
        "buffer_sizes": [3, 2, 1, 3, 5, 8, 1],
        "instructions": [
            {"type": "COPY_MASKED", "input": 0, "output": 1, "indexes": [0, 2]},
            {"type": "COPY_MASKED", "input": 0, "output": 2, "indexes": [1]},
            {
                "type": "MAP_TRANSFORM", "input": 2, "output": 3, "internal_input_index": 0,
                "internal_output_index": 0, "map": 0, "size": 3,
            },
            copy(1, 4, 0),
            copy(3, 4, 2),
            dot(4, 5, 0, "RELU"),
            dot(5, 6, 1),
        ],
    }

    encoded = B.MultiIdEmbeddings([1], [[1, 2]], [4])(inp)
    model = B.ModelGraph(inp, B.Dense(1)(B.Dense(8, activation="relu")(encoded)))
    train(model, x, y, epochs=20)
    im, _ = roundtrip(model, x, features)
    assert (len(im["weights"]), len(im["bias"]), len(im["maps"])) == (2, 2, 1)
    assert im["maps"][0].keys() == {1, 2}
    assert len(im["instructions"][1].pop("default")) == 4
    assert structure(im) == {
        "features": features,
        "buffer_sizes": [3, 6, 8, 1],
        "instructions": [
            {"type": "COPY_MASKED", "input": 0, "output": 1, "indexes": [0, 2]},
            {
                "type": "MAP_TRANSFORM", "input": 0, "output": 1, "internal_input_index": 1,
                "internal_output_index": 2, "map": 0, "size": 4,
            },
            dot(1, 2, 0, "RELU"),
            dot(2, 3, 1),
        ],
    }


def test_embeddings_reject_duplicate_and_default_ids(B):
    with pytest.raises(ValueError, match="Duplicate ID"):
        B.SingleIdEmbeddings([1, 1], 2)
    with pytest.raises(ValueError, match="default_id"):
        B.SingleIdEmbeddings([1, -1], 2)


def test_scale_and_shift_vectorized(B):
    inp = B.InputBuffer(3)
    scaled = B.ScaleVectorized([2.0, 0.5, 3.0], in_place=True)(inp)
    model = B.ModelGraph(inp, B.ShiftVectorized([1.0, -1.0, 0.0], in_place=True)(scaled))
    x = np.array([[1.0, 2.0, 3.0], [0.0, -4.0, 2.0]])

    im, pred = roundtrip(model, x)
    assert np.allclose(pred, x * [2.0, 0.5, 3.0] + [1.0, -1.0, 0.0], atol=1e-6)
    assert im["parameters"] == [[2.0, 0.5, 3.0], [1.0, -1.0, 0.0]]
    assert structure(im) == {
        "features": [],
        "buffer_sizes": [3],
        "instructions": [
            {"type": "MUL_ELEMENTWISE", "input": 0, "parameters": 0},
            {"type": "ADD_ELEMENTWISE", "input": 0, "parameters": 1},
        ],
    }


def test_scale_and_shift_not_inplace(B):
    inp = B.InputBuffer(2)
    scaled = B.ScaleVectorized([2.0, 3.0], in_place=False)(inp)
    model = B.ModelGraph(inp, B.ShiftVectorized([10.0, 20.0], in_place=False)(scaled))
    assert structure(model.create_instruction_model()) == {
        "features": [],
        "buffer_sizes": [2, 2, 2],
        "instructions": [
            copy(0, 1, 0),
            {"type": "MUL_ELEMENTWISE", "input": 1, "parameters": 0},
            copy(1, 2, 0),
            {"type": "ADD_ELEMENTWISE", "input": 2, "parameters": 1},
        ],
    }


@pytest.mark.parametrize("op, value, fn", [("ScaleVectorized", 2.0, np.multiply), ("ShiftVectorized", 5.0, np.add)])
def test_scalar_constant_broadcasts(B, op, value, fn):
    inp = B.InputBuffer(4)
    model = B.ModelGraph(inp, getattr(B, op)(value, in_place=True)(inp))
    x = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 0.0, 2.0]])
    im, pred = roundtrip(model, x)
    assert im["parameters"] == [[value] * 4]
    assert np.allclose(pred, fn(x, value), atol=1e-6)


def test_reduce_sum(B):
    inp = B.InputBuffer(5)
    model = B.ModelGraph(inp, B.ReduceSum()(inp))
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 1.5, 2.5, 3.5, 4.5]], dtype=np.float32)
    assert np.allclose(model.predict_on_batch(x), [[15.0], [12.5]], atol=1e-6)
    im, _ = roundtrip(model, x)
    assert im["buffer_sizes"] == [5, 1]
    assert im["instructions"] == [{"type": "REDUCE_SUM", "input": 0, "output": 1}]

    inp = B.InputBuffer(4)
    model = B.ModelGraph(inp, B.ReduceSum()(B.Dense(8, activation="relu")(inp)))
    _, pred = roundtrip(model, np.random.randn(10, 4).astype(np.float32))
    assert pred.shape == (10, 1)


@pytest.mark.parametrize("activation", ["gelu", "softplus"])
def test_dense_activation_trains_and_exports(B, activation):
    inp = B.InputBuffer(4)
    h = B.Dense(8, activation=activation)(inp)
    model = B.ModelGraph(inp, B.Dense(1, activation="sigmoid")(h))
    x = np.random.random((10, 4)).astype(np.float32)
    train(model, x, np.random.randint(0, 2, size=(10, 1)), loss="binary_crossentropy")
    im, _ = roundtrip(model, x)
    assert [i["activation"] for i in im["instructions"]] == [activation.upper(), "SIGMOID"]

    code = {"gelu": "g", "softplus": "p"}[activation]
    ff = B.ff_model([4, 8, 1], B.NO_BATCH_NORM, code + "s").create_instruction_model()
    assert [i["activation"] for i in ff["instructions"]] == [activation.upper(), "SIGMOID"]

    model = B.ModelGraph(inp, B.Dense(3)(h))
    roundtrip(model, np.random.randn(20, 4).astype(np.float32) * 5)


ACTIVATION_INPUT = np.array(
    [[-100.0, -88.0, -20.0, -2.0, -1.0, -0.01, 0.0, 0.01, 0.5, 1.0, 2.0, 20.0, 88.0, 100.0]], dtype=np.float32
)


@pytest.mark.parametrize(
    "activation",
    ["relu", "sigmoid", "tanh", "softmax", "gelu", "softplus", "sqrt", "log", "log10", "inverse", "exp", "sign"],
)
def test_activation_computation_matches_instruction_model(B, activation):
    inp = B.InputBuffer(ACTIVATION_INPUT.shape[1])
    model = B.ModelGraph(inp, B.ActivationComputation(activation, in_place=False)(inp))
    im, pred = roundtrip(model, ACTIVATION_INPUT)
    assert im["instructions"][-1] == {"type": "ACTIVATION", "input": 1, "activation": activation.upper()}
    if activation == "exp":
        assert pred[0, 0] == pred[0, 1] and pred[0, -1] == pred[0, -2]  # clamped to [-88, 88]
        assert np.allclose(pred, np.exp(np.clip(ACTIVATION_INPUT, -88, 88)), rtol=1e-6)
    if activation == "sign":
        assert np.array_equal(pred, np.sign(ACTIVATION_INPUT))


def test_activation_in_place_reuses_buffer(B):
    inp = B.InputBuffer(3)
    model = B.ModelGraph(inp, B.ActivationComputation("tanh", in_place=True)(inp))
    im, _ = roundtrip(model, np.random.randn(4, 3).astype(np.float32))
    assert im["buffer_sizes"] == [3]
    assert im["instructions"] == [{"type": "ACTIVATION", "input": 0, "activation": "TANH"}]


@pytest.mark.parametrize(
    "op, instr, heads, data, expected",
    [
        ("MultiplyHeads", "MULTIPLY_BUFFER_HEADS", [[2.0, 3.0, 4.0, 5.0]], [[1.0] * 20], [[2.0] * 5 + [3.0] * 5 + [4.0] * 5 + [5.0] * 5]),
        ("AddHeads", "ADD_BUFFER_HEADS", [[1.0, 2.0, 3.0, 4.0]], [[0.0] * 20], [[1.0] * 5 + [2.0] * 5 + [3.0] * 5 + [4.0] * 5]),
        ("MultiplyHeads", "MULTIPLY_BUFFER_HEADS", [[2.0]], [list(range(1, 11))], [[2.0 * i for i in range(1, 11)]]),
    ],
)
def test_heads(B, op, instr, heads, data, expected):
    data, heads = np.array(data, dtype=float), np.array(heads)
    d_buf, h_buf = B.InputBuffer(data.shape[1]), B.InputBuffer(heads.shape[1])
    model = B.ModelGraph([d_buf, h_buf], getattr(B, op)()([d_buf, h_buf]))
    pred = model.predict([data, heads], verbose=0)
    assert np.allclose(pred, expected, atol=1e-6)
    im = model.create_instruction_model()
    assert im["instructions"][-1]["type"] == instr
    assert np.allclose(instruction_model_inference(im, [data, heads])[-1], pred, atol=1e-6)


def test_heads_reject_indivisible_sizes(B):
    with pytest.raises(ValueError, match="divisible"):
        B.MultiplyHeads()([B.InputBuffer(5), B.InputBuffer(2)])
    with pytest.raises(ValueError, match="exactly two inputs"):
        B.AddHeads()([B.InputBuffer(4)])


def test_multiply_buffers(B):
    a, b = B.InputBuffer(3), B.InputBuffer(3)
    model = B.ModelGraph([a, b], B.Multiply()([a, b]))
    x = [np.random.randn(4, 3), np.random.randn(4, 3)]
    im = model.create_instruction_model()
    assert im["instructions"] == [{"type": "MULTIPLY_ELEMENTWISE_BUFFERS", "input": [0, 1], "output": 2}]
    assert np.allclose(instruction_model_inference(im, x)[-1], model.predict(x, verbose=0), atol=1e-6)
    with pytest.raises(ValueError, match="at least two"):
        B.Multiply()([a])


@pytest.mark.parametrize(
    "kwargs, params, bounds",
    [
        ({"clip_min": [0.0, -1.0, 0.5], "clip_max": [1.0, 2.0, 3.0]}, [[0.0, -1.0, 0.5], [1.0, 2.0, 3.0]], ("min", "max")),
        ({"clip_min": [0.0, -1.0, 0.5]}, [[0.0, -1.0, 0.5]], ("min",)),
        ({"clip_max": [1.0, 2.0, 3.0]}, [[1.0, 2.0, 3.0]], ("max",)),
        ({"clip_min": 0.0, "clip_max": 1.0}, [[0.0] * 3, [1.0] * 3], ("min", "max")),
    ],
)
def test_clip_vectorized(B, kwargs, params, bounds):
    inp = B.InputBuffer(3)
    model = B.ModelGraph(inp, B.ClipVectorized(in_place=True, **kwargs)(inp))
    x = np.array([[-2.0, 5.0, 1.0], [0.5, -3.0, 4.0], [0.2, 0.3, 0.9]])
    im, pred = roundtrip(model, x)
    assert np.allclose(pred, np.clip(x, kwargs.get("clip_min"), kwargs.get("clip_max")), atol=1e-6)
    assert im["parameters"] == params
    assert structure(im) == {
        "features": [],
        "buffer_sizes": [3],
        "instructions": [{"type": "CLIP_ELEMENTWISE", "input": 0, **{f"parameters_{b}": i for i, b in enumerate(bounds)}}],
    }


def test_clip_vectorized_not_inplace(B):
    inp = B.InputBuffer(2)
    model = B.ModelGraph(inp, B.ClipVectorized(clip_min=0.0, clip_max=1.0, in_place=False)(inp))
    assert structure(model.create_instruction_model()) == {
        "features": [],
        "buffer_sizes": [2, 2],
        "instructions": [
            copy(0, 1, 0),
            {"type": "CLIP_ELEMENTWISE", "input": 1, "parameters_min": 0, "parameters_max": 1},
        ],
    }


def test_clip_vectorized_no_bounds_raises(B):
    with pytest.raises(ValueError, match="at least one"):
        B.ClipVectorized()


def test_clip_vectorized_in_pipeline(B):
    inp = B.InputBuffer(4)
    clipped = B.ClipVectorized(clip_min=0.0, clip_max=5.0, in_place=True)(B.Dense(8, activation="relu")(inp))
    model = B.ModelGraph(inp, B.Dense(1, activation="sigmoid")(clipped))
    x = np.random.random((10, 4)).astype(np.float32)
    train(model, x, np.random.randint(0, 2, size=(10, 1)).astype(np.float32), loss="binary_crossentropy")
    roundtrip(model, x)


def test_validation_data_and_scoring(B):
    model = B.ff_model([3, 4, 1], B.INPLACE, "rs")
    im = model.create_instruction_model()
    im["validation_data"] = B.generate_validation_data(["a", "b", "c"], model, means=[1, 2, 3], stds=[1, 1, 2])
    validate_instruction_model(im)

    x = np.random.randn(30, 3).astype(np.float32)
    y = np.random.randn(30)
    assert B.score_selection(model, x, y) == pytest.approx(B.score_selection(im, x, y), abs=1e-6)
    assert B.tau_compare(np.c_[y, -y], np.c_[y, y]) == pytest.approx([1.0, -1.0])


def test_torch_fit_history_and_sample_weight():
    torch_backend = pytest.importorskip("instmodel.torch")
    model = torch_backend.ff_model([3, 4, 1], torch_backend.NO_BATCH_NORM, "rl")
    model.compile(optimizer="sgd", loss="mse", lr=0.05)
    x, y = np.random.randn(64, 3), np.random.randn(64, 1)
    history = model.fit(x, y, epochs=3, batch_size=16, verbose=0, sample_weight=np.ones(64))
    assert len(history) == 3 and all(np.isfinite(history))
    with pytest.raises(ValueError, match="Unknown loss"):
        model.compile(loss="nope")
