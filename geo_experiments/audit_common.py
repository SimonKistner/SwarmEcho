"""Pure NumPy history fingerprints shared by current and historical audits."""
from __future__ import annotations
import hashlib
import numpy as np

STAGES = ("Raw observation", "Encoded observation", "GRU input (encoding + received context)")

def digest(*arrays):
    """Exact float32 fingerprints, treating signed zero as equal like np.unique."""
    result = hashlib.sha256()
    for array in arrays:
        array = np.array(array, copy=True, order="C")
        if array.dtype.kind == "f":
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite features found; history comparisons would be invalid.")
            array[array == 0] = 0
        result.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        result.update(array.tobytes())
    return result.digest()


def cross_cell_pairs(counts):
    total = sum(counts.values())
    return (total * (total - 1) - sum(n * (n - 1) for n in counts.values())) // 2


def differences(a, b):
    delta = np.abs(a - b)
    return (float(delta.max()), float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
            float((delta == 0).mean()), float(np.all(delta == 0, axis=1).mean()))


