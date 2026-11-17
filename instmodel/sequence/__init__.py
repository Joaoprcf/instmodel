"""Causal transformers: training on PyTorch or TensorFlow, token-by-token inference from JSON."""

import importlib

from ._spec import (  # noqa: F401
    FORMAT_VERSION,
    SegmentFeatures,
    TransformerConfig,
    check_parameters,
    export_step_model,
    init_parameters,
    parameter_count,
    parameter_shapes,
    segment_features,
)
from .reference import (  # noqa: F401
    SequenceSession,
    add_validation_sequences,
    new_state,
    validate_sequence_model,
)


def __getattr__(name):
    # Training backends are optional and heavy; import them on first access.
    if name in ("tf", "torch"):
        try:
            return importlib.import_module(f".{name}", __name__)
        except ImportError as e:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r} ({e})") from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
