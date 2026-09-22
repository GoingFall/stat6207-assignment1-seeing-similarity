"""Readers for TexMex fvecs/ivecs files with strict shape validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_vecs(path: Path, dtype: np.dtype, expected_dimension: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Empty vector file: {path}")
    dimension = int(raw[0])
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(f"{path}: dimension {dimension}, expected {expected_dimension}")
    width = dimension + 1
    if raw.size % width:
        raise ValueError(f"{path}: byte count is inconsistent with dimension {dimension}")
    matrix = raw.reshape(-1, width)
    if not np.all(matrix[:, 0] == dimension):
        raise ValueError(f"{path}: inconsistent row dimensions")
    payload = matrix[:, 1:]
    if np.dtype(dtype) == np.dtype(np.float32):
        return payload.view(np.float32).copy()
    return payload.astype(dtype, copy=True)


def read_fvecs(path: Path, expected_dimension: int | None = None) -> np.ndarray:
    return read_vecs(path, np.float32, expected_dimension)


def read_ivecs(path: Path, expected_dimension: int | None = None) -> np.ndarray:
    return read_vecs(path, np.int32, expected_dimension)
