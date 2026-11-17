"""NumPy reference for sequence models: token-by-token steps with explicit state.

This is the ground truth the training backends and the Rust runtime are
checked against. It runs the exported instruction list once per token on
float32 arrays, for several sequences at a time.
"""

import numpy as np

from ..instruction_model import instruction_model_inference, model_arrays


def new_state(model, sequences=1):
    """Empty state for ``sequences`` parallel sequences: attention caches, scanners and position."""
    info = model["sequence"]
    capacity = info["max_positions"]

    def cache(slot):
        return np.zeros((sequences, capacity, slot["kv_heads"], slot["head_dim"]), dtype=np.float32)

    def scanner():
        return dict(prev_member=np.zeros(sequences, dtype=bool), prev_run=np.zeros(sequences, dtype=bool),
                    index=np.zeros(sequences, dtype=np.int64), ordinal=np.zeros(sequences, dtype=np.int64))

    return dict(position=0, kv=[dict(keys=cache(slot), values=cache(slot)) for slot in info["kv_slots"]],
                scanners=[scanner() for _ in range(info["scanners"])])


class SequenceSession:
    """Steps a format-v2 sequence model; ``state`` may be copied, kept and restored by the caller."""

    def __init__(self, model, sequences=1):
        if model.get("format_version") != 2 or "sequence" not in model:
            raise ValueError("not a format-v2 sequence model")
        self.model, self.sequences = model, sequences
        self._arrays = model_arrays(model)
        self.state = new_state(model, sequences)

    def step(self, tokens):
        """Feeds one token id per sequence and returns next-token logits ``[sequences, vocabulary]``."""
        tokens = np.asarray(tokens, dtype=np.float32).reshape(self.sequences, 1)
        logits = instruction_model_inference(self.model, [tokens], self.state, self._arrays)[-1]
        self.state["position"] += 1
        return logits

    def run(self, tokens):
        """Logits ``[sequences, time, vocabulary]`` for whole token arrays ``[sequences, time]``."""
        tokens = np.asarray(tokens).reshape(self.sequences, -1)
        return np.stack([self.step(tokens[:, t]) for t in range(tokens.shape[1])], 1)


def add_validation_sequences(model, sequences):
    """Stores token sequences with the reference logits, for runtimes to replay on load."""
    model["validation_sequences"] = [
        dict(tokens=[int(t) for t in tokens], logits=SequenceSession(model).run(tokens)[0].tolist())
        for tokens in sequences
    ]
    return model


def validate_sequence_model(model, atol=1e-4):
    """Replays ``validation_sequences``; returns the largest logit error or raises on a mismatch."""
    worst = 0.0
    for case in model.get("validation_sequences", []):
        expected = np.asarray(case["logits"], dtype=np.float32)
        actual = SequenceSession(model).run(case["tokens"])[0]
        worst = max(worst, float(np.abs(actual - expected).max()))
        if worst > atol or (actual.argmax(-1) != expected.argmax(-1)).any():
            raise AssertionError(f"sequence model differs from its validation logits (max error {worst})")
    return worst
