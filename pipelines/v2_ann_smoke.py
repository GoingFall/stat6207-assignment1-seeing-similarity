"""Correctness and resource smoke tests for native Windows CPU Faiss."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np

from pipelines.prepare_data import ROOT
from src.benchmarking.resources import ResourceMonitor, process_rss_bytes


def recall_at_k(expected: np.ndarray, actual: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(left[:k]) & set(right[:k])) / k for left, right in zip(expected, actual)]))


def measure(index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    index.search(queries[: min(32, len(queries))], k)
    latencies = []
    with ResourceMonitor(0.1) as monitor:
        for query in queries:
            start = time.perf_counter()
            distances, ids = index.search(query[None, :], k)
            latencies.append((time.perf_counter() - start) * 1000)
    return distances, ids, {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "qps": 1000.0 / float(np.mean(latencies)),
        "resources": monitor.summary(),
    }


def index_size(index) -> int:
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as handle:
        path = Path(handle.name)
    try:
        faiss.write_index(index, str(path))
        return path.stat().st_size
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    config = json.loads((ROOT / "configs/v2.0/experiment.json").read_text())
    rng = np.random.default_rng(config["seed"])
    database = rng.standard_normal((10000, 128), dtype=np.float32)
    queries = rng.standard_normal((200, 128), dtype=np.float32)
    k = 10
    faiss.omp_set_num_threads(16)

    exact_reference = faiss.IndexFlatL2(128)
    exact_reference.add(database)
    exact_distances, exact_ids = exact_reference.search(queries, k)
    records = []

    definitions = [
        ("flat", faiss.IndexFlatL2(128)),
        ("hnsw", faiss.IndexHNSWFlat(128, config["ann"]["hnsw"]["M"])),
        ("ivf_flat", faiss.IndexIVFFlat(faiss.IndexFlatL2(128), 128, 256)),
        ("ivf_pq", faiss.IndexIVFPQ(faiss.IndexFlatL2(128), 128, 256, 16, 8)),
    ]
    for name, index in definitions:
        before = process_rss_bytes()
        start = time.perf_counter()
        if name == "hnsw":
            index.hnsw.efConstruction = config["ann"]["hnsw"]["efConstruction"]
        if not index.is_trained:
            index.train(database)
        index.add(database)
        if name == "hnsw":
            index.hnsw.efSearch = 128
        if name.startswith("ivf"):
            index.nprobe = 16
        build_seconds = time.perf_counter() - start
        distances, ids = index.search(queries, k)
        assert ids.shape == exact_ids.shape and np.all(ids >= 0)
        _, _, timing = measure(index, queries[:50], k)
        records.append({
            "index": name,
            "build_seconds": build_seconds,
            "rss_delta_bytes": max(0, process_rss_bytes() - before),
            "serialized_bytes": index_size(index),
            "recall_at_1": recall_at_k(exact_ids, ids, 1),
            "recall_at_10": recall_at_k(exact_ids, ids, 10),
            "timing": timing,
            "distance_shape": list(distances.shape),
        })

    result = {
        "status": "passed",
        "faiss": getattr(faiss, "__version__", "unknown"),
        "threads": faiss.omp_get_max_threads(),
        "database_shape": list(database.shape),
        "query_shape": list(queries.shape),
        "exact_first_distance": float(exact_distances[0, 0]),
        "indexes": records,
    }
    folder = ROOT / "results/v2.0/gates"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "faiss_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
