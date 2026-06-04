from __future__ import annotations

import numpy as np


def minmax_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    denom = max(max_value - min_value, eps)
    return (array - min_value) / denom


def infer_square_grid(num_patches: int) -> tuple[int, int]:
    side = int(round(num_patches ** 0.5))
    if side * side != num_patches:
        raise ValueError(f"num_patches must be a square number, got {num_patches}.")
    return side, side
