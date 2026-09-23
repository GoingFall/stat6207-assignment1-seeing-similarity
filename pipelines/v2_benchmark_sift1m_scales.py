"""Measure preregistered SIFT scale curves with parameters frozen by the 1M development set."""
from __future__ import annotations

import gc
import hashlib
import json
import tempfile
import time
from pathlib import Path

import faiss

from pipelines.prepare_data import ROOT
from pipelines.v2_benchmark_sift1m import build_index
from src.benchmarking.ann import recall_metrics, timed_search
from src.benchmarking.resources import ResourceMonitor, process_rss_bytes
from src.benchmarking.vector_io import read_fvecs
from src.paths import resolve


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    benchmark_path = ROOT / "results/v2.0/ann/sift1m.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if benchmark["status"] != "passed":
        raise RuntimeError("Passed 1M development selection is required")
    config = json.loads((ROOT / "configs/v2.0/experiment.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "data/v2.0/manifests/sift1m_lock.json").read_text(encoding="utf-8"))
    base = read_fvecs(resolve(lock["files"]["base"]["path"]))
    queries = read_fvecs(resolve(lock["files"]["query"]["path"]))
    learn = read_fvecs(resolve(lock["files"]["learn"]["path"]))
    development = queries[: config["ann"]["query_counts"]["development"]]
    held_out = queries[config["ann"]["query_counts"]["development"] :]
    k = config["exact"]["k"]
    repetitions = config["ann"]["timed_repetitions"]
    faiss.omp_set_num_threads(16)
    result = {
        "status": "running",
        "selection_source": "results/v2.0/ann/sift1m.json development queries at 1M scale; no per-scale retuning",
        "selection_source_sha256": hash_file(benchmark_path),
        "scales": {},
    }
    for scale in config["ann"]["scales"]:
        if scale == len(base):
            result["scales"][str(scale)] = {"source": "results/v2.0/ann/sift1m.json", "families": {
                family: {"params": data["selected_params"], "test": data["test"]}
                for family, data in benchmark["families"].items()
            }}
            continue
        vectors = base[:scale]
        exact = faiss.IndexFlatL2(vectors.shape[1])
        exact.add(vectors)
        _, development_truth = exact.search(development, k)
        _, held_out_truth = exact.search(held_out, k)
        _, exact_batch = timed_search(exact, held_out, k, 256, repetitions)
        scale_result = {"exact_test": exact_batch, "families": {}, "skipped": {}}
        del exact
        gc.collect()
        with tempfile.TemporaryDirectory(prefix=f"stat6207-sift-{scale}-") as folder:
            for family, family_result in benchmark["families"].items():
                params = family_result["selected_params"]
                if family.startswith("ivf") and params["nlist"] * 39 > min(len(learn), scale):
                    scale_result["skipped"][family] = {
                        "reason": "fixed selected nlist lacks 39 training vectors per centroid at this scale",
                        "params": params,
                    }
                    continue
                index = build_index(family, vectors.shape[1], params)
                before = process_rss_bytes()
                start = time.perf_counter()
                with ResourceMonitor(0.2) as monitor:
                    if not index.is_trained:
                        index.train(learn[: min(len(learn), scale)])
                    index.add(vectors)
                build = {
                    "seconds": time.perf_counter() - start, "rss_before_bytes": before,
                    "rss_after_bytes": process_rss_bytes(), "resources": monitor.summary(),
                }
                if family == "hnsw":
                    index.hnsw.efSearch = params["efSearch"]
                else:
                    index.nprobe = params["nprobe"]
                path = Path(folder) / f"{family}.faiss"
                faiss.write_index(index, str(path))
                development_ids, development_timing = timed_search(index, development, k, 256, repetitions)
                held_out_ids, held_out_timing = timed_search(index, held_out, k, 256, repetitions)
                _, single_timing = timed_search(index, held_out[:1000], k, 1, 1)
                scale_result["families"][family] = {
                    "params": params, "build": build, "serialized_bytes": path.stat().st_size,
                    "development": {**recall_metrics(development_truth, development_ids), "batch_256": development_timing},
                    "test": {**recall_metrics(held_out_truth, held_out_ids), "batch_256": held_out_timing, "single": single_timing},
                }
                del index
                gc.collect()
        result["scales"][str(scale)] = scale_result
        output = ROOT / "results/v2.0/ann/sift1m_scales_partial.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["status"] = "passed"
    output = ROOT / "results/v2.0/ann/sift1m_scales.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "scales": list(result["scales"])}, indent=2))


if __name__ == "__main__":
    main()
