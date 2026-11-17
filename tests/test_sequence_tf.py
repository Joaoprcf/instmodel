import numpy as np
import pytest

from instmodel import sequence
from test_sequence import CONFIGS, randomized, tokens_for

tf = st = None


@pytest.fixture(scope="module", autouse=True)
def backend():
    """Imports the framework only when a test of this file runs.

    Collecting this file must not load it: the two frameworks cannot share a
    process on every installation, and ``-k`` deselects one backend's tests.
    """
    global tf, st
    tf = pytest.importorskip("tensorflow")
    tf.config.experimental.enable_tensor_float_32_execution(False)  # The checks below are float32-exact.
    from instmodel.sequence import tf as st


def config_of(name):
    return sequence.TransformerConfig(**CONFIGS[name])


def logits_of(model, tokens):
    return model(tf.constant(tokens)).numpy()


@pytest.mark.parametrize("name", CONFIGS)
def test_forward_matches_the_exported_step_model(name):
    config = config_of(name)
    model = st.Transformer(config, half=False).load_parameters(randomized(config))
    tokens = tokens_for(config)
    stepped = sequence.SequenceSession(sequence.export_step_model(config, model.member_parameters()),
                                       len(tokens)).run(tokens)
    np.testing.assert_allclose(logits_of(model, tokens[None])[0], stepped, atol=2e-5)
    again = st.Transformer(config, seed=5).load_parameters(model.member_parameters()).member_parameters()
    for key, value in randomized(config).items():
        np.testing.assert_array_equal(again[key], value)


@pytest.mark.parametrize("name", ["grouped", "looped"])
def test_members_are_independent(name):
    config = config_of(name)
    model = st.Transformer(config, members=3, half=False)
    for member in range(3):
        model.load_parameters(randomized(config, seed=member), member=member)
    tokens = tokens_for(config)
    together = logits_of(model, np.stack([tokens] * 3))
    for member in range(3):
        single = model.select(member)
        np.testing.assert_array_equal(single.member_parameters()["embedding"],
                                      randomized(config, seed=member)["embedding"])
        np.testing.assert_allclose(logits_of(single, tokens[None])[0], together[member], atol=1e-5)  # Kernel order.
    assert np.abs(together[0] - together[1]).max() > 1e-2


@pytest.mark.parametrize("name", CONFIGS)
def test_initialization_follows_the_spec(name):
    config = config_of(name)
    model = st.Transformer(config, members=2, seed=3)
    expected = sequence.init_parameters(config, 3, 2)
    actual = {key: v.numpy() for key, v in model.named_parameters()}
    assert list(actual) == list(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
    assert model.parameter_count == sequence.parameter_count(config)


def slow_loss(logits, targets, groups, groups_count, allowed=None, soft=()):
    """Per-row loss and flags, one position at a time."""
    lead = targets.shape[:-1]
    logits, targets, groups = (a.reshape(-1, *a.shape[len(lead):]) for a in (logits, targets, groups))
    soft = [a.reshape(-1, *a.shape[len(lead):]) for a in soft]
    loss, wrong, follows = [], [], []
    for r in range(len(targets)):
        sums, counts, bad, agree = np.zeros(groups_count), np.zeros(groups_count), False, True
        forks = {}
        if soft:
            for slot, position in enumerate(soft[0][r]):
                if position >= 0:
                    forks[int(position)] = (soft[1][r, slot], soft[2][r, slot])
        for t, group in enumerate(groups[r]):
            row = logits[r, t].astype(np.float64)
            if allowed is not None:
                row = np.where(allowed[max(group, 0)], row, -np.inf)
            logp = row - np.log(np.exp(row - row.max()).sum()) - row.max()
            if t in forks:
                tokens, shares = forks[t]
                value = -sum(share * logp[token] for token, share in zip(tokens, shares) if share > 0)
                agree &= row.argmax() == targets[r, t]
            elif targets[r, t] == -100:
                value = 0.0
            else:
                value = -logp[targets[r, t]]
                bad |= row.argmax() != targets[r, t]
            if group >= 0:
                sums[group] += value
                counts[group] += 1
        loss.append((sums / np.maximum(counts, 1)).mean())
        wrong.append(bad)
        follows.append(agree)
    return [np.reshape(v, lead) for v in (loss, wrong, follows)]


def loss_case(seed=0, vocab=7, groups_count=3, time=9):
    rng = np.random.default_rng(seed)
    groups = rng.integers(-1, groups_count, (2, 3, time))
    groups[..., 4], groups[..., 6] = 2, 1  # Soft positions sit in groups 2 and 1.
    groups[0, 0] = np.where(groups[0, 0] == 0, 2, groups[0, 0])  # Group 0 absent from one row.
    allowed = rng.random((groups_count, vocab)) < 0.6
    allowed[:, :2] = True
    legal = np.array([[[rng.choice(np.flatnonzero(allowed[max(g, 0)])) for g in row] for row in m] for m in groups])
    tokens = np.concatenate((rng.integers(0, vocab, (2, 3, 1)), legal), -1)
    positions = np.array([4, -1])[None, None].repeat(2, 0).repeat(3, 1)
    positions[1, :, 1] = 6
    positions[0, 2] = -1  # A row without soft targets.
    soft_tokens = np.zeros((2, 3, 2, 3), dtype=np.int64)
    soft_tokens[..., :2] = [0, 1]
    shares = np.zeros((2, 3, 2, 3), dtype=np.float32)
    shares[..., :2] = [0.7, 0.3]
    logits = rng.normal(0, 2, (2, 3, time, vocab)).astype(np.float32)
    return logits, tokens, groups, allowed, (positions, soft_tokens, shares)


def test_sequence_loss_matches_a_per_position_computation():
    logits, tokens, groups, allowed, soft = loss_case()
    x, y, g = st.batch_targets(tf.constant(tokens), tf.constant(groups))
    assert (x.numpy() == tokens[..., :-1]).all() and ((y.numpy() == -100) == (groups < 0)).all()
    tensors = [tf.constant(a) for a in soft]
    for mask, extra in ((allowed, ()), (allowed, soft), (None, soft)):
        result = st.sequence_loss(tf.constant(logits), y, g, None if mask is None else tf.constant(mask),
                                  *(tensors if extra else ()), groups_count=4 if mask is None else None)
        expected = slow_loss(logits, y.numpy(), g.numpy(), 3 if mask is not None else 4, mask, extra)
        np.testing.assert_allclose(result[0].numpy(), expected[0], rtol=1e-5)
        for flags, reference in zip(result[1:], expected[1:]):
            np.testing.assert_array_equal(flags.numpy(), reference)


def test_sequence_loss_flags():
    targets = np.array([[1, 2, 3, -100], [1, 2, 3, 0], [1, 2, 3, 0]])
    groups = tf.constant([[0, 0, 1, -1], [0, 0, 1, 1], [0, 0, 1, 1]])
    logits = np.eye(5, dtype=np.float32)[np.maximum(targets, 0)] * 5
    logits[1, 3] = [0., 5., 0., 0., 0.]  # A mistaken action.
    logits[2, 1] = [0., 0., 0., 0., 5.]  # A soft position choosing another branch.
    positions = tf.constant([[1, -1], [1, -1], [1, -1]])
    soft_tokens = tf.constant([[[2, 4]] * 2] * 3)
    shares = tf.constant([[[0.5, 0.5]] * 2] * 3)
    loss, wrong, follows = st.sequence_loss(tf.constant(logits), tf.constant(targets), groups, None, positions,
                                            soft_tokens, shares, groups_count=2)
    assert wrong.numpy().tolist() == [False, True, False] and follows.numpy().tolist() == [True, True, False]
    assert np.isfinite(loss.numpy()).all()


def test_training_learns_a_successor_rule():
    rng = np.random.default_rng(0)
    vocab, length, rows = 6, 8, 64
    config = sequence.TransformerConfig(vocab_size=vocab, width=16, heads=2, layers=1)
    start = np.arange(rows) % vocab
    tokens = (start[:, None] + np.arange(length + 1)) % vocab
    soft_positions = np.where(start[:, None] % 2 == 0, 0, -1)  # Half the rows learn step 0 as a soft target.
    data = dict(tokens=tokens, groups=np.zeros((rows, length), dtype=np.int64), weight=rng.random(rows) + 0.5,
                soft_positions=soft_positions, soft_tokens=((start + 1) % vocab)[:, None, None],
                soft_shares=np.ones((rows, 1, 1), dtype=np.float32))
    trainer = st.Trainer(st.Transformer(config, members=2), data, batch=32, lr=1e-2, ema=0.9)  # XLA-compiled.
    steps = 300
    for step in range(steps):
        trainer.step(st.schedule(step / steps, 1e-2))
    assert trainer.loss.shape == (2,) and (trainer.loss.numpy() < 0.05).all()
    model = trainer.averaged().select(1)
    generated = np.arange(vocab)[:, None]
    for _ in range(length):
        generated = np.concatenate((generated, logits_of(model, generated[None])[0, :, -1].argmax(-1)[:, None]), 1)
    np.testing.assert_array_equal(generated, tokens[:vocab])
    session = sequence.SequenceSession(sequence.export_step_model(config, model.member_parameters()), vocab)
    stepped = [np.arange(vocab)]
    for _ in range(length):
        stepped.append(session.step(stepped[-1]).argmax(-1))
    np.testing.assert_array_equal(np.stack(stepped, 1), generated)


def test_optimizer_step_matches_numpy_adamw():
    config, rows = config_of("looped"), 6
    rng = np.random.default_rng(2)
    tokens = rng.integers(0, config.vocab_size, (rows, 10))
    groups = rng.integers(-1, 2, (rows, 9))
    groups[:, 3] = 1  # The soft position's group.
    data = dict(tokens=tokens, groups=groups, soft_positions=np.full((rows, 1), 3), soft_tokens=tokens[:, None, 4:6],
                soft_shares=np.array([[[0.6, 0.4]]], dtype=np.float32).repeat(rows, 0))
    model = st.Transformer(config, members=2, half=False)
    for member in range(2):
        model.load_parameters(randomized(config, seed=member), member=member)
    lr, betas, decay, ema = 3e-3, (0.8, 0.95), 0.1, 0.7
    trainer = st.Trainer(model, data, batch=4, lr=lr, ema=ema, betas=betas, weight_decay=decay)
    for key, value in randomized(config, seed=1).items():  # The compiling update is discarded.
        np.testing.assert_array_equal(model.member_parameters(1)[key], value)
    names = [name for name, _ in model.named_parameters()]
    params = {name: v.numpy().astype(np.float64) for name, v in model.named_parameters()}
    average = {name: value.copy() for name, value in params.items()}
    moments = [{name: np.zeros_like(value) for name, value in params.items()} for _ in range(2)]
    for step, index in enumerate(([[0, 1, 2, 3], [5, 4, 3, 2]], [[1, 1, 5, 0], [2, 3, 0, 4]]), 1):
        index = np.array(index)
        x, y, g = st.batch_targets(tokens[index], groups[index])
        with tf.GradientTape() as tape:
            loss = st.sequence_loss(model(x), y, g, None, *(data[k][index] for k in st.SOFT), groups_count=2)[0]
            total = tf.reduce_sum(tf.reduce_mean(loss, -1))
        grads = {n: tf.convert_to_tensor(q).numpy().astype(np.float64)
                 for n, q in zip(names, tape.gradient(total, [v for _, v in model.named_parameters()]))}
        norms = np.sqrt(sum((q.reshape(2, -1) ** 2).sum(-1) for q in grads.values()))
        scale = np.minimum(1 / (norms + 1e-6), 1)
        assert (scale < 1).any()  # The clipping is exercised.
        # Adam turns float32 noise in a vanishing gradient (the key bias) into a full step, so check the rest.
        keep, checked, total = {}, 0, 0
        for n in names:
            q = grads[n] * scale.reshape(-1, *[1] * (grads[n].ndim - 1))
            keep[n] = np.abs(q) > 1e-5
            checked, total = checked + keep[n].sum(), total + q.size
            params[n] *= 1 - lr * decay
            moments[0][n] = betas[0] * moments[0][n] + (1 - betas[0]) * q
            moments[1][n] = betas[1] * moments[1][n] + (1 - betas[1]) * q * q
            denom = np.sqrt(moments[1][n]) / np.sqrt(1 - betas[1] ** step) + 1e-8
            params[n] -= lr / (1 - betas[0] ** step) * moments[0][n] / denom
            average[n] += (1 - ema) * (params[n] - average[n])
        assert checked > 0.85 * total
        trainer._step_rows(tf.constant(index))
        np.testing.assert_allclose(trainer.loss.numpy(), tf.reduce_mean(loss, -1).numpy(), rtol=1e-5)
        for n, v, a in zip(names, trainer.parameters, trainer.average):
            np.testing.assert_allclose(v.numpy()[keep[n]], params[n][keep[n]], atol=2e-6, err_msg=n)
            np.testing.assert_allclose(a.numpy()[keep[n]], average[n][keep[n]], atol=2e-6, err_msg=n)
        params = {n: v.numpy().astype(np.float64) for n, v in zip(names, trainer.parameters)}  # No drift.
        average = {n: a.numpy().astype(np.float64) for n, a in zip(names, trainer.average)}
    assert trainer.steps.numpy() == 2


def test_lower_bound_matches_searchsorted():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 64, 1000):
        ascending = np.sort(rng.integers(0, 20, n)).astype(np.float64)
        values = rng.integers(-1, 22, (3, 50)).astype(np.float64)
        compiled = tf.function(st.lower_bound, jit_compile=True)
        np.testing.assert_array_equal(compiled(tf.constant(ascending), tf.constant(values)).numpy(),
                                      np.searchsorted(ascending, values))


def test_several_steps_per_call_equal_single_steps():
    config = config_of("grouped")
    rng = np.random.default_rng(0)
    data = dict(tokens=rng.integers(0, 14, (32, 13)), groups=rng.integers(-1, 2, (32, 12)))
    finals = []
    for calls in ((1, 1, 1), (3,)):
        trainer = st.Trainer(st.Transformer(config, members=2, half=False), data, batch=4, lr=1e-2)
        for steps in calls:
            trainer.step(1e-2, steps)
        assert float(trainer.steps.numpy()) == 3
        finals.append(trainer.model.member_parameters(1))
    keys = slice(config.width, config.width + config.kv_heads * config.head_dim)
    for key, value in finals[0].items():
        if key.endswith("qkv_bias"):  # Adam steps the key bias on rounding noise: its true gradient is zero.
            value, finals[1][key] = np.delete(value, keys), np.delete(finals[1][key], keys)
        np.testing.assert_allclose(finals[1][key], value, atol=1e-6, err_msg=key)


def test_schedule_endpoints():
    assert st.schedule(0, 1.0) == 0
    assert st.schedule(0.03, 1.0) == pytest.approx(1.0)
    assert st.schedule(1, 1.0) == pytest.approx(0.02)
    assert st.schedule(0.5, 2.0, warmup=0.0, floor=0.0) == pytest.approx(1.0)
