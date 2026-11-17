import copy
import json

import numpy as np
import pytest

from instmodel import sequence
from instmodel.instruction_model import concatenate_instruction_models, instruction_model_inference

FEATURES = sequence.SegmentFeatures(members=range(10), openers=[11], rows=6)
CONFIGS = {
    "plain": dict(vocab_size=14, width=8, heads=2),
    "features": dict(vocab_size=14, width=8, heads=2, features=FEATURES),
    "grouped": dict(vocab_size=14, width=12, heads=4, kv_heads=2, ff_width=10, features=FEATURES),
    "untied": dict(vocab_size=14, width=8, heads=4, kv_heads=1, tied=False, layers=1),
    "looped": dict(vocab_size=14, width=8, heads=2, layers=1, loops=3, inject=True, pass_embed=True,
                   pass_norms=True, lora=2, features=FEATURES),
    "stacked_loops": dict(vocab_size=14, width=8, heads=2, layers=2, loops=2, pass_embed=True),
}


def randomized(config, seed=0):
    """Parameters with every array random, so zero-initialized pieces are exercised too."""
    rng = np.random.default_rng(seed)
    return {name: (value + rng.normal(0, 0.3, value.shape)).astype(np.float32)
            for name, value in sequence.init_parameters(config, seed).items()}


def forward(config, p, tokens):
    """Independent whole-sequence forward for one token row, written from the model definition."""
    c, d, time = config, config.width, len(tokens)

    def norm(x, values):
        x = x - x.mean(-1, keepdims=True)
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + c.epsilon) * values[0] + values[1]

    def gelu(x):
        from scipy.special import erf
        return x * 0.5 * (1 + erf(x / np.sqrt(2)))

    source = p["embedding"][tokens]
    if c.features is not None:
        ordinal, index = sequence.segment_features(tokens, c.features)
        source = source + p["ordinal"][ordinal] + p["index"][index]
    x, head, group = source, c.head_dim, c.heads // c.kv_heads
    blocked = np.triu(np.ones((time, time), dtype=bool), 1)
    for step in range(c.loops):
        if step and c.inject:
            x = x + source
        if c.pass_embed:
            x = x + p["passes"][step]
        row = step if c.pass_norms else 0
        for layer in range(c.layers):
            b = {name.split(".")[-1]: value for name, value in p.items() if name.startswith(f"blocks.{layer}.")}
            hidden = norm(x, b["norm1"][row])
            qkv = hidden @ b["qkv"] + b["qkv_bias"]
            if c.lora:
                qkv = qkv + hidden @ b["lora_a"][step] @ b["lora_b"][step]
            q = qkv[:, :d].reshape(time, c.heads, head)
            k = qkv[:, d:d + c.kv_heads * head].reshape(time, c.kv_heads, head)
            v = qkv[:, d + c.kv_heads * head:].reshape(time, c.kv_heads, head)
            mixed = np.zeros((time, c.heads, head))
            for h in range(c.heads):
                scores = q[:, h] @ k[:, h // group].T / np.sqrt(head)
                scores[blocked] = -np.inf
                scores = np.exp(scores - scores.max(-1, keepdims=True))
                mixed[:, h] = scores / scores.sum(-1, keepdims=True) @ v[:, h // group]
            x = x + mixed.reshape(time, d) @ b["out"] + b["out_bias"]
            x = x + gelu(norm(x, b["norm2"][row]) @ b["up"] + b["up_bias"]) @ b["down"] + b["down_bias"]
    readout = p["embedding"].T if c.tied else p["readout"]
    return norm(x, p["norm"]) @ readout + p["readout_bias"]


def tokens_for(config, rows=3, time=17, seed=1):
    return np.random.default_rng(seed).integers(0, config.vocab_size, (rows, time))


@pytest.mark.parametrize("name", CONFIGS)
def test_steps_match_the_sequence_definition(name):
    config = sequence.TransformerConfig(**CONFIGS[name])
    parameters = randomized(config)
    model = json.loads(json.dumps(sequence.export_step_model(config, parameters)))
    tokens = tokens_for(config)
    stepped = sequence.SequenceSession(model, len(tokens)).run(tokens)
    for row, expected in zip(stepped, (forward(config, parameters, t) for t in tokens)):
        np.testing.assert_allclose(row, expected, atol=2e-5)
    assert len(model["sequence"]["kv_slots"]) == config.layers * config.loops
    assert sequence.parameter_count(config) == sum(v.size for v in parameters.values())


def test_shared_weights_are_stored_once():
    config = sequence.TransformerConfig(**CONFIGS["looped"])
    model = sequence.export_step_model(config, randomized(config))
    by_type = {}
    for instruction in model["instructions"]:
        by_type.setdefault(instruction["type"], []).append(instruction)
    table = by_type["EMBEDDING"][0]["weights"]
    assert model["instructions"][-1] == dict(type="DOT", input=len(model["buffer_sizes"]) - 2,
                                             output=len(model["buffer_sizes"]) - 1, weights=table)
    assert model["bias"][table] == pytest.approx(randomized(config)["readout_bias"].tolist())
    wide = [i["weights"] for i in by_type["DOT"] if model["buffer_sizes"][i["output"]] == config.qkv_size
            and model["buffer_sizes"][i["input"]] == config.width]
    assert len(wide) == config.loops and len(set(wide)) == 1  # One qkv matrix serves every pass.
    norms = [i["gain"] for i in by_type["LAYER_NORM"]]
    assert len(set(norms)) == len(norms)  # Per-pass norms stay distinct.
    assert [i["slot"] for i in by_type["CAUSAL_ATTENTION"]] == list(range(config.loops))


def test_segment_features_follow_signed_runs():
    features = sequence.SegmentFeatures(members=range(10), openers=[11], rows=4)
    text = [11, 1, 2, 11, 11, 3, 12, 4, 5, 6, 7, 8, 11, 12, 9]  # -12--3=45678-=9
    ordinal, index = sequence.segment_features(text, features)
    assert index.tolist() == [1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 3, 3, 0, 0, 1]
    assert ordinal.tolist() == [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    with pytest.raises(ValueError):
        sequence.SegmentFeatures(members=[1], openers=[1])


def test_state_can_be_kept_restored_and_batched():
    config = sequence.TransformerConfig(**CONFIGS["grouped"])
    model = sequence.export_step_model(config, randomized(config))
    tokens = tokens_for(config, rows=2, time=12)
    whole = sequence.SequenceSession(model, 2).run(tokens)
    session = sequence.SequenceSession(model, 2)
    session.run(tokens[:, :5])
    kept = copy.deepcopy(session.state)
    session.run(tokens_for(config, rows=2, time=4, seed=9))  # A discarded continuation.
    session.state = kept
    np.testing.assert_array_equal(session.run(tokens[:, 5:]), whole[:, 5:])
    np.testing.assert_allclose(sequence.SequenceSession(model).run(tokens[1])[0], whole[1], atol=1e-5)


def test_validation_sequences_detect_changed_weights():
    config = sequence.TransformerConfig(**CONFIGS["features"])
    model = sequence.export_step_model(config, randomized(config), vocabulary=[f"t{i}" for i in range(14)],
                                       metadata=dict(note="kept"))
    sequence.add_validation_sequences(model, tokens_for(config, rows=2).tolist())
    assert sequence.validate_sequence_model(json.loads(json.dumps(model))) < 1e-6
    assert model["sequence"]["metadata"] == dict(note="kept") and model["format_version"] == 2
    model["parameters"][0][0] += 0.5
    with pytest.raises(AssertionError):
        sequence.validate_sequence_model(model)


def test_invalid_use_is_rejected():
    config = sequence.TransformerConfig(vocab_size=14, width=8, heads=2, max_positions=3)
    model = sequence.export_step_model(config, randomized(config))
    session = sequence.SequenceSession(model)
    session.run([1, 2, 3])
    with pytest.raises(ValueError, match="max_positions"):
        session.step([1])
    with pytest.raises(ValueError, match="EMBEDDING id"):
        sequence.SequenceSession(model).step([14])
    with pytest.raises(ValueError, match="needs sequence state"):
        instruction_model_inference(model, np.zeros((1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="cannot be concatenated"):
        concatenate_instruction_models([model, model])
    with pytest.raises(ValueError, match="vocabulary length"):
        sequence.export_step_model(config, randomized(config), vocabulary=["a"])
    with pytest.raises(ValueError, match="parameter names"):
        sequence.export_step_model(config, {})
    broken = copy.deepcopy(model)
    next(i for i in broken["instructions"] if i["type"] == "CAUSAL_ATTENTION")["slot"] = 9
    with pytest.raises(ValueError, match="slot index"):
        sequence.SequenceSession(broken)
    for bad in (dict(width=9), dict(heads=4, kv_heads=3), dict(loops=0)):
        with pytest.raises(ValueError):
            sequence.TransformerConfig(**{**CONFIGS["plain"], **bad})
