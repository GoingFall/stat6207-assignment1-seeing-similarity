"""Run the preregistered native-CPU SIFT1M exact and ANN benchmark."""
from __future__ import annotations

import gc
import hashlib
import json
import tempfile
import time
from pathlib import Path

import faiss

from pipelines.prepare_data import ROOT
from src.benchmarking.ann import choose_candidate, recall_metrics, sharded_exact_l2, timed_search
from src.benchmarking.resources import ResourceMonitor, process_rss_bytes
from src.benchmarking.vector_io import read_fvecs, read_ivecs


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def serialize(index, folder: Path, name: str) -> tuple[Path, int]:
    path = folder / f"{name}.faiss"
    faiss.write_index(index, str(path))
    return path, path.stat().st_size


def build_index(kind: str, dimension: int, params: dict):
    if kind == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, params["M"])
        index.hnsw.efConstruction = params["efConstruction"]
        return index
    quantizer = faiss.IndexFlatL2(dimension)
    if kind == "ivf_flat":
        return faiss.IndexIVFFlat(quantizer, dimension, params["nlist"])
    if kind == "ivf_pq":
        return faiss.IndexIVFPQ(quantizer, dimension, params["nlist"], params["m"], params["nbits"])
    raise ValueError(kind)


def main() -> None:
    config_path = ROOT / "configs/v2/experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ann = config["ann"]
    lock = json.loads((ROOT / "data_v2/manifests/sift1m_lock.json").read_text(encoding="utf-8"))
    arrays = {
        name: (read_ivecs if name == "groundtruth" else read_fvecs)(ROOT / record["path"])
        for name, record in lock["files"].items()
    }
    base, queries, learn, truth = arrays["base"], arrays["query"], arrays["learn"], arrays["groundtruth"]
    dev_count = ann["query_counts"]["development"]
    dev_queries, test_queries = queries[:dev_count], queries[dev_count:]
    dev_truth, test_truth = truth[:dev_count], truth[dev_count:]
    k = ann["exact"]["k"] if "exact" in ann else config["exact"]["k"]
    repetitions = ann["timed_repetitions"]
    faiss.omp_set_num_threads(16)
    output = ROOT / "results_v2/ann"
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "config_sha256": hash_file(config_path),
        "ann_config": ann,
        "ann_config_sha256": canonical_hash(ann),
        "exact_config": config["exact"],
        "exact_config_sha256": canonical_hash(config["exact"]),
        "sift1m_config": config["sift1m"],
        "sift1m_config_sha256": canonical_hash(config["sift1m"]),
        "data_lock_sha256": hash_file(ROOT / "data_v2/manifests/sift1m_lock.json"),
        "faiss": getattr(faiss, "__version__", "unknown"),
        "threads": faiss.omp_get_max_threads(),
        "metric_definition": "mean set overlap against official exact top-k; equal-distance tie ordering may reduce overlap without distance error",
        "development_queries": dev_count,
        "test_queries": len(test_queries),
        "families": {},
    }

    # Authoritative exhaustive baseline and supplied-ground-truth validation.
    exact = faiss.IndexFlatL2(base.shape[1])
    before = process_rss_bytes()
    start = time.perf_counter()
    with ResourceMonitor(0.2) as monitor:
        exact.add(base)
    result["exact_build"] = {
        "seconds": time.perf_counter() - start,
        "rss_before_bytes": before,
        "rss_after_bytes": process_rss_bytes(),
        "resources": monitor.summary(),
    }
    feasibility_queries = queries[: config["exact"]["feasibility_queries"]]
    start = time.perf_counter()
    with ResourceMonitor(0.2) as monitor:
        _, exact_ids = sharded_exact_l2(
            base,
            feasibility_queries,
            k,
            config["exact"]["query_batch"],
            config["exact"]["database_shard"],
        )
    elapsed = time.perf_counter() - start
    exact_timing = {
        "seconds": elapsed,
        "qps": len(feasibility_queries) / elapsed,
        "query_batch": config["exact"]["query_batch"],
        "database_shard": config["exact"]["database_shard"],
        "resources": monitor.summary(),
    }
    exact_metrics = recall_metrics(truth[: len(exact_ids)], exact_ids)
    if exact_metrics["recall_at_100"] < 0.9999:
        raise AssertionError(f"Sharded exact materially disagrees with official ground truth: {exact_metrics}")
    result["exact_feasibility"] = {"implementation": "bounded NumPy exact L2 with deterministic shard top-k merge", **exact_metrics, **exact_timing}
    del exact
    gc.collect()

    grids = {
        "hnsw": [
            {"M": ann["hnsw"]["M"], "efConstruction": ann["hnsw"]["efConstruction"], "efSearch": value}
            for value in ann["hnsw"]["efSearch"]
        ],
        "ivf_flat": [
            {"nlist": nlist, "nprobe": nprobe}
            for nlist in ann["ivf_flat"]["nlist"] if nlist * 39 <= len(learn)
            for nprobe in ann["ivf_flat"]["nprobe"] if nprobe <= nlist
        ],
        "ivf_pq": [
            {"nlist": nlist, "nprobe": nprobe, "m": ann["ivf_pq"]["m"], "nbits": ann["ivf_pq"]["nbits"]}
            for nlist in ann["ivf_pq"]["nlist"] if nlist * 39 <= len(learn)
            for nprobe in ann["ivf_pq"]["nprobe"] if nprobe <= nlist
        ],
    }
    result["ivf_feasibility"] = {
        "training_vectors": len(learn),
        "minimum_vectors_per_centroid": 39,
        "excluded_nlist": sorted({
            nlist
            for family in ("ivf_flat", "ivf_pq")
            for nlist in ann[family]["nlist"]
            if nlist * 39 > len(learn)
        }),
    }
    with tempfile.TemporaryDirectory(prefix="stat6207-ann-") as temporary:
        temporary = Path(temporary)
        for family, candidates in grids.items():
            grouped = {}
            for params in candidates:
                structural = tuple((key, value) for key, value in params.items() if key not in ("efSearch", "nprobe"))
                grouped.setdefault(structural, []).append(params)
            records = []
            saved = {}
            for structural, variants in grouped.items():
                build_params = dict(structural)
                index = build_index(family, base.shape[1], build_params | variants[0])
                before = process_rss_bytes()
                start = time.perf_counter()
                with ResourceMonitor(0.2) as monitor:
                    if not index.is_trained:
                        index.train(learn)
                    index.add(base)
                build = {
                    "seconds": time.perf_counter() - start,
                    "rss_before_bytes": before,
                    "rss_after_bytes": process_rss_bytes(),
                    "resources": monitor.summary(),
                }
                path, size = serialize(index, temporary, family + "-" + "-".join(f"{k}{v}" for k, v in structural))
                saved[structural] = path
                for params in variants:
                    if family == "hnsw":
                        index.hnsw.efSearch = params["efSearch"]
                    else:
                        index.nprobe = params["nprobe"]
                    dev_ids, batch_timing = timed_search(index, dev_queries, k, 256, repetitions)
                    _, single_timing = timed_search(index, dev_queries[: min(1000, len(dev_queries))], k, 1, 1)
                    name = family + "-" + "-".join(f"{key}{value}" for key, value in sorted(params.items()))
                    records.append({
                        "name": name,
                        "params": params,
                        "build": build,
                        "serialized_bytes": size,
                        "development": {**recall_metrics(dev_truth, dev_ids), "batch_256": batch_timing, "single": single_timing},
                    })
                del index
                gc.collect()
            selected = choose_candidate(records, ann["selection"]["minimum_development_recall_at_10"])
            structural = tuple((key, value) for key, value in selected["params"].items() if key not in ("efSearch", "nprobe"))
            index = faiss.read_index(str(saved[structural]))
            if family == "hnsw":
                index.hnsw.efSearch = selected["params"]["efSearch"]
            else:
                index.nprobe = selected["params"]["nprobe"]
            test_ids, batch_timing = timed_search(index, test_queries, k, 256, repetitions)
            _, single_timing = timed_search(index, test_queries[:1000], k, 1, 1)
            result["families"][family] = {
                "selection_rule": ann["selection"],
                "development_candidates": records,
                "selected": selected["name"],
                "selected_params": selected["params"],
                "test": {**recall_metrics(test_truth, test_ids), "batch_256": batch_timing, "single": single_timing},
            }
            (output / "sift1m_partial.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            del index
            gc.collect()

    result["status"] = "passed"
    (output / "sift1m.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "exact_feasibility": result["exact_feasibility"],
        "selected": {family: data["selected_params"] for family, data in result["families"].items()},
        "test": {family: {key: value for key, value in data["test"].items() if key.startswith("recall_")} for family, data in result["families"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
