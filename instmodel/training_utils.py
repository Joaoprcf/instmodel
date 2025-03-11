from decimal import Decimal
import numpy as np
import pandas as pd

def hide_id(datax, id_k):
    datax = datax.copy()
    datax[id_k] = -1.0
    return datax

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return [convert_to_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return Decimal(str(float(obj)))  # Convert floats to Decimal for precision
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    return obj
