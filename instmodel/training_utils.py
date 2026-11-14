from decimal import Decimal

import numpy as np


def hide_id(datax, id_k):
    """Copy of datax with column id_k set to -1.0 (the unknown-id bucket)."""
    datax = datax.copy()
    datax[id_k] = -1.0
    return datax


def convert_to_serializable(obj):
    """Recursively converts NumPy arrays/scalars to lists, ints and Decimal (exact float repr)."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return [convert_to_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return Decimal(str(float(obj)))
    if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    return obj
