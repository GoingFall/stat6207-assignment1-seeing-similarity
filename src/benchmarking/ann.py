"""ANN metrics and repeatable search timing helpers."""
from __future__ import annotations

import time

import numpy as np

from src.benchmarking.resources import ResourceMonitor


def sharded_exact_l2(
    database: np.ndarray,
    queries: np.ndarray,
    k: int,
    query_batch: int,
    database_shard: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact squared-L2 top-k using bounded query/database blocks and deterministic merging."""
    if database.dtype != np.float32 or queries.dtype != np.float32:
        raise ValueError("Exact search inputs must be float32")
    all_distances = []
    all_ids = []
    for query_begin in range(0, len(queries), query_batch):
        query = queries[query_begin : query_begin + query_batch]
        best_distances = np.full((len(query), k), np.inf, dtype=np.float32)
        best_ids = np.full((len(query), k), -1, dtype=np.int64)
        for database_begin in range(0, len(database), database_shard):
            shard = database[database_begin : database_begin + database_shard]
            # Algebraic L2 avoids allocating a query-by-database-by-dimension tensor.
            distances = (
                np.sum(query * query, axis=1, keepdims=True)
                + np.sum(shard * shard, axis=1)[None, :]
                - 2.0 * (query @ shard.T)
            )
            np.maximum(distances, 0.0, out=distances)
            shard_k = min(k, len(shard))
            positions = np.argpartition(distances, shard_k - 1, axis=1)[:, :shard_k]
            shard_distances = np.take_along_axis(distances, positions, axis=1)
            shard_ids = positions.astype(np.int64) + database_begin
            merged_distances = np.concatenate((best_distances, shard_distances), axis=1)
            merged_ids = np.concatenate((best_ids, shard_ids), axis=1)
            keep = np.argpartition(merged_distances, k - 1, axis=1)[:, :k]
            best_distances = np.take_along_axis(merged_distances, keep, axis=1)
            best_ids = np.take_along_axis(merged_ids, keep, axis=1)
        # Lexicographic sorting makes distance ties deterministic by vector ID.
        for row in range(len(query)):
            order = np.lexsort((best_ids[row], best_distances[row]))
            best_distances[row] = best_distances[row, order]
            best_ids[row] = best_ids[row, order]
        all_distances.append(best_distances)
        all_ids.append(best_ids)
    return np.concatenate(all_distances), np.concatenate(all_ids)


def overlap_recall(expected: np.ndarray, actual: np.ndarray, k: int) -> float:
    """Mean top-k set overlap, the ANN recall definition used in this project."""
    if expected.shape[0] != actual.shape[0] or expected.shape[1] < k or actual.shape[1] < k:
        raise ValueError("Expected and actual neighbours must contain at least k entries per query")
    total = 0
    for truth, found in zip(expected[:, :k], actual[:, :k]):
        total += len(set(map(int, truth)) & set(map(int, found)))
    return total / (len(expected) * k)


def recall_metrics(expected: np.ndarray, actual: np.ndarray) -> dict:
    return {f"recall_at_{k}": overlap_recall(expected, actual, k) for k in (1, 10, 100)}


def timed_search(index, queries: np.ndarray, k: int, batch_size: int, repetitions: int) -> tuple[np.ndarray, dict]:
    if len(queries) == 0:
        raise ValueError("Cannot benchmark an empty query set")
    index.search(queries[: min(batch_size, len(queries))], k)
    durations = []
    last_ids = None
    with ResourceMonitor(0.2) as monitor:
        for _ in range(repetitions):
            chunks = []
            for begin in range(0, len(queries), batch_size):
                batch = queries[begin : begin + batch_size]
                start = time.perf_counter()
                _, ids = index.search(batch, k)
                durations.append(time.perf_counter() - start)
                chunks.append(ids)
            last_ids = np.concatenate(chunks)
    per_query_ms = np.array(durations) * 1000 / np.array(
        [min(batch_size, len(queries) - begin) for _ in range(repetitions) for begin in range(0, len(queries), batch_size)]
    )
    elapsed = float(sum(durations))
    return last_ids, {
        "batch_size": batch_size,
        "repetitions": repetitions,
        "batches": len(durations),
        "search_seconds": elapsed,
        "qps": len(queries) * repetitions / elapsed,
        "per_query_p50_ms": float(np.percentile(per_query_ms, 50)),
        "per_query_p95_ms": float(np.percentile(per_query_ms, 95)),
        "per_query_p99_ms": float(np.percentile(per_query_ms, 99)),
        "resources": monitor.summary(),
    }


def choose_candidate(records: list[dict], minimum_recall: float) -> dict:
    eligible = [record for record in records if record["development"]["recall_at_10"] >= minimum_recall]
    if eligible:
        return min(eligible, key=lambda record: (record["development"]["batch_256"]["per_query_p95_ms"], -record["development"]["recall_at_10"], record["name"]))
    return min(records, key=lambda record: (-record["development"]["recall_at_10"], record["development"]["batch_256"]["per_query_p95_ms"], record["name"]))
