# MIT License
# Copyright (c) 2025 João Ferreira
# See LICENSE file for details.
import importlib

from . import instruction_model, training_utils  # noqa: F401


def __getattr__(name):
    # Training backends are optional and heavy; import them on first access.
    if name in ("tf", "torch"):
        try:
            return importlib.import_module(f".{name}", __name__)
        except ImportError as e:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r} ({e})") from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
