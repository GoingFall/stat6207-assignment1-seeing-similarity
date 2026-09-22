"""Retrieval metrics with exact image-ID self exclusion."""
from __future__ import annotations

import numpy as np


def remove_self(query_ids: np.ndarray, neighbour_ids: np.ndarray) -> np.ndarray:
    if len(query_ids) != len(neighbour_ids):
        raise ValueError("Query and neighbour rows differ")
    output = np.empty((len(query_ids), neighbour_ids.shape[1] - 1), dtype=neighbour_ids.dtype)
    for row, (query_id, neighbours) in enumerate(zip(query_ids, neighbour_ids)):
        kept = neighbours[neighbours != query_id]
        if len(kept) < output.shape[1]:
            raise AssertionError(f"Insufficient non-self neighbours for query {query_id}")
        output[row] = kept[: output.shape[1]]
    return output


def retrieval_metrics(product_ids: np.ndarray, neighbours: np.ndarray, ks=(1, 10, 100), map_k=1000) -> dict:
    if neighbours.shape[1] < max(max(ks), map_k):
        raise ValueError("Neighbour matrix is too short for requested metrics")
    relevant = product_ids[neighbours] == product_ids[:, None]
    result = {f"recall_at_{k}": float(np.mean(np.any(relevant[:, :k], axis=1))) for k in ks}
    cumulative = np.cumsum(relevant[:, :map_k], axis=1)
    ranks = np.arange(1, map_k + 1)[None, :]
    positives = np.bincount(product_ids)[product_ids] - 1
    denominator = np.minimum(positives, map_k)
    if np.any(denominator <= 0):
        raise AssertionError("Every query must have another same-product image")
    average_precision = np.sum((cumulative / ranks) * relevant[:, :map_k], axis=1) / denominator
    result[f"map_at_{map_k}"] = float(np.mean(average_precision))
    return result
