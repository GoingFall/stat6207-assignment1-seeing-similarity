"""Strict controlled long-tail sampling with immutable selected IDs."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np


def class_count_schedule(class_ids: Sequence[int], maximum: int, ratio: int) -> dict[int, int]:
    """Create a deterministic geometric schedule with an exact integer endpoint ratio."""
    classes = sorted(map(int, class_ids))
    if len(classes) < 2 or maximum <= 0 or ratio <= 1:
        raise ValueError("Need at least two classes, positive maximum and ratio > 1")
    if maximum % ratio:
        raise ValueError(
            f"Exact {ratio}:1 integer ratio is infeasible with maximum count {maximum}; "
            "do not round, replace samples, or borrow validation/test images"
        )
    minimum = maximum // ratio
    raw = np.geomspace(maximum, minimum, num=len(classes))
    counts = np.clip(np.rint(raw).astype(int), minimum, maximum)
    counts[0], counts[-1] = maximum, minimum
    return dict(zip(classes, map(int, counts)))


def sample_ids(
    candidates: Mapping[int, Sequence[int]],
    ratio: int,
    seed: int,
) -> dict:
    classes = sorted(map(int, candidates))
    lengths = {class_id: len(candidates[class_id]) for class_id in classes}
    if len(set(lengths.values())) != 1:
        raise ValueError("Controlled comparison requires an equal candidate count per class")
    maximum = next(iter(lengths.values()))
    counts = class_count_schedule(classes, maximum, ratio)
    selected = {}
    for class_id in classes:
        values = np.array(sorted(map(int, candidates[class_id])), dtype=np.int64)
        rng = np.random.default_rng(seed + class_id)
        chosen = np.sort(rng.choice(values, size=counts[class_id], replace=False))
        selected[str(class_id)] = list(map(int, chosen))
    minimum = maximum // ratio
    result = {
        "ratio": ratio,
        "seed": seed,
        "maximum_count": maximum,
        "minimum_count": minimum,
        "realized_ratio": maximum / minimum,
        "counts": {str(key): value for key, value in counts.items()},
        "selected_image_ids": selected,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["sha256_without_hash_field"] = hashlib.sha256(payload).hexdigest()
    return result
