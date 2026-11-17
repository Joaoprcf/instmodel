"""Writes small sequence models with validation logits, for other runtimes to replay.

Usage: python tests/make_sequence_fixtures.py OUTPUT_DIRECTORY
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import test_sequence as cases  # noqa: E402

from instmodel import sequence  # noqa: E402


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    for name, values in cases.CONFIGS.items():
        config = sequence.TransformerConfig(**{**values, "max_positions": 24})
        model = sequence.export_step_model(config, cases.randomized(config))
        sequence.add_validation_sequences(model, cases.tokens_for(config, rows=2, time=20).tolist())
        (output / f"{name}.json").write_text(json.dumps(model, separators=(",", ":")))
        print(name, len(model["instructions"]), "instructions")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
