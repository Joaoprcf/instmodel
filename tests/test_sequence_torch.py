import numpy as np
import pytest

torch = pytest.importorskip("torch")

from instmodel import sequence  # noqa: E402
from instmodel.sequence import torch as st  # noqa: E402
from test_sequence import CONFIGS, randomized, tokens_for  # noqa: E402


def config_of(name):
    return sequence.TransformerConfig(**CONFIGS[name])


def logits_of(model, tokens):
    with torch.no_grad():
        return model(torch.as_tensor(tokens).long()).numpy()


@pytest.mark.parametrize("name", CONFIGS)
def test_forward_matches_the_exported_step_model(name):
    config = config_of(name)
    model = st.Transformer(config, bf16=False).load_parameters(randomized(config))
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
    model = st.Transformer(config, members=3, bf16=False)
    for member in range(3):
        model.load_parameters(randomized(config, seed=member), member=member)
    tokens = tokens_for(config)
    together = logits_of(model, np.stack([tokens] * 3))
    for member in range(3):
        single = model.select(member)
        np.testing.assert_array_equal(single.member_parameters()["embedding"],
                                      randomized(config, seed=member)["embedding"])
        np.testing.assert_allclose(logits_of(single, tokens[None])[0], together[member], atol=1e-6)
    assert np.abs(together[0] - together[1]).max() > 1e-2


@pytest.mark.parametrize("name", CONFIGS)
def test_initialization_follows_the_spec(name):
    config = config_of(name)
    model = st.Transformer(config, members=2, seed=3)
    expected = sequence.init_parameters(config, 3, 2)
    actual = {key: p.detach().numpy() for key, p in model.named_parameters()}
    assert actual.keys() == expected.keys()
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
    x, y, g = st.batch_targets(torch.as_tensor(tokens), torch.as_tensor(groups))
    assert (x.numpy() == tokens[..., :-1]).all() and ((y.numpy() == -100) == (groups < 0)).all()
    tensors = [torch.as_tensor(a) for a in soft]
    for mask, extra in ((allowed, ()), (allowed, soft), (None, soft)):
        result = st.sequence_loss(torch.as_tensor(logits), y, g, None if mask is None else torch.as_tensor(mask),
                                  *(tensors if extra else ()), groups_count=4 if mask is None else None)
        expected = slow_loss(logits, y.numpy(), g.numpy(), 3 if mask is not None else 4, mask, extra)
        np.testing.assert_allclose(result[0].numpy(), expected[0], rtol=1e-5)
        for flags, reference in zip(result[1:], expected[1:]):
            np.testing.assert_array_equal(flags.numpy(), reference)


def test_sequence_loss_flags():
    targets = torch.tensor([[1, 2, 3, -100], [1, 2, 3, 0], [1, 2, 3, 0]])
    groups = torch.tensor([[0, 0, 1, -1], [0, 0, 1, 1], [0, 0, 1, 1]])
    logits = torch.nn.functional.one_hot(targets.clamp_min(0), 5).float() * 5
    logits[1, 3] = torch.tensor([0., 5., 0., 0., 0.])  # A mistaken action.
    logits[2, 1] = torch.tensor([0., 0., 0., 0., 5.])  # A soft position choosing another branch.
    positions = torch.tensor([[1, -1], [1, -1], [1, -1]])
    soft_tokens = torch.tensor([[[2, 4]] * 2] * 3)
    shares = torch.tensor([[[0.5, 0.5]] * 2] * 3)
    loss, wrong, follows = st.sequence_loss(logits, targets, groups, None, positions, soft_tokens, shares,
                                            groups_count=2)
    assert wrong.tolist() == [False, True, False] and follows.tolist() == [True, True, False]
    assert torch.isfinite(loss).all()


def test_training_learns_a_successor_rule():
    torch.manual_seed(0)
    vocab, length, rows = 6, 8, 64
    config = sequence.TransformerConfig(vocab_size=vocab, width=16, heads=2, layers=1)
    start = torch.arange(rows) % vocab
    tokens = (start[:, None] + torch.arange(length + 1)) % vocab
    soft_positions = torch.where(start[:, None] % 2 == 0, 0, -1)  # Half the rows learn step 0 as a soft target.
    data = dict(tokens=tokens, groups=torch.zeros(rows, length, dtype=torch.long), weight=torch.rand(rows) + 0.5,
                soft_positions=soft_positions, soft_tokens=((start + 1) % vocab)[:, None, None],
                soft_shares=torch.ones(rows, 1, 1))
    trainer = st.Trainer(st.Transformer(config, members=2), data, batch=32, lr=1e-2, ema=0.9)
    steps = 300
    for step in range(steps):
        trainer.step(st.schedule(step / steps, 1e-2))
    assert trainer.loss.shape == (2,) and (trainer.loss < 0.05).all()
    model = trainer.averaged().select(1)
    generated = torch.arange(vocab)[:, None]
    with torch.no_grad():
        for _ in range(length):
            generated = torch.cat((generated, model(generated[None])[0, :, -1].argmax(-1, keepdim=True)), 1)
    np.testing.assert_array_equal(generated.numpy(), tokens[:vocab].numpy())
    session = sequence.SequenceSession(sequence.export_step_model(config, model.member_parameters()), vocab)
    stepped = [np.arange(vocab)]
    for _ in range(length):
        stepped.append(session.step(stepped[-1]).argmax(-1))
    np.testing.assert_array_equal(np.stack(stepped, 1), generated.numpy())


def test_schedule_endpoints():
    assert st.schedule(0, 1.0) == 0
    assert st.schedule(0.03, 1.0) == pytest.approx(1.0)
    assert st.schedule(1, 1.0) == pytest.approx(0.02)
    assert st.schedule(0.5, 2.0, warmup=0.0, floor=0.0) == pytest.approx(1.0)
