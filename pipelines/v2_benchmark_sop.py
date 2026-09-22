"""Evaluate standard SOP leave-one-out retrieval and CPU ANN trade-offs."""
from __future__ import annotations

import gc
import hashlib
import json
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np

from checks.v2_contracts import assert_no_self_match
from pipelines.prepare_data import ROOT
from src.benchmarking.ann import choose_candidate, overlap_recall, timed_search
from src.benchmarking.resources import ResourceMonitor, process_rss_bytes
from src.retrieval.metrics import remove_self, retrieval_metrics


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def ann_recall(truth: np.ndarray, actual: np.ndarray) -> dict:
    return {f"recall_at_{k}": overlap_recall(truth, actual, k) for k in (1, 10, 100)}


def build(family: str, dimension: int, params: dict):
    if family == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, params["M"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = params["efConstruction"]
        return index
    quantizer = faiss.IndexFlatIP(dimension)
    if family == "ivf_flat":
        return faiss.IndexIVFFlat(quantizer, dimension, params["nlist"], faiss.METRIC_INNER_PRODUCT)
    return faiss.IndexIVFPQ(quantizer, dimension, params["nlist"], params["m"], params["nbits"], faiss.METRIC_INNER_PRODUCT)


def main() -> None:
    config_path = ROOT / "configs/v2/experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ann, sop = config["ann"], config["sop"]
    lock_path = ROOT / "data_v2/manifests/sop_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = load_manifest(ROOT / lock["manifests"]["test"]["path"])
    vectors_path = ROOT / "data_v2/embeddings/sop/dinov2_test.npy"
    vectors = np.load(vectors_path)
    if vectors.shape != (len(manifest), 384):
        raise AssertionError("Embedding rows do not match the frozen SOP test manifest")
    query_ids = np.arange(len(vectors), dtype=np.int64)
    product_ids = np.array([row["product_id"] for row in manifest], dtype=np.int64)
    k_search = 1001
    faiss.omp_set_num_threads(16)
    output = ROOT / "results_v2/sop"
    output.mkdir(parents=True, exist_ok=True)

    exact = faiss.IndexFlatIP(vectors.shape[1])
    before = process_rss_bytes()
    start = time.perf_counter()
    with ResourceMonitor(0.2) as monitor:
        exact.add(vectors)
    exact_build = {"seconds": time.perf_counter() - start, "rss_before_bytes": before, "rss_after_bytes": process_rss_bytes(), "resources": monitor.summary()}
    exact_ids, exact_timing = timed_search(exact, vectors, k_search, 256, 1)
    exact_ids = remove_self(query_ids, exact_ids)
    assert_no_self_match(query_ids, exact_ids)
    semantic = retrieval_metrics(product_ids, exact_ids, ks=(1, 10, 100), map_k=1000)
    np.save(output / "exact_top1000.npy", exact_ids, allow_pickle=False)

    development = np.arange(0, len(vectors), 6, dtype=np.int64)[:10000]
    test_mask = np.ones(len(vectors), dtype=bool)
    test_mask[development] = False
    held_out = np.flatnonzero(test_mask)
    grids = {
        "hnsw": [
            {"M": ann["hnsw"]["M"], "efConstruction": ann["hnsw"]["efConstruction"], "efSearch": value}
            for value in ann["hnsw"]["efSearch"]
        ],
        "ivf_flat": [
            {"nlist": nlist, "nprobe": nprobe}
            for nlist in ann["ivf_flat"]["nlist"] if nlist * 39 <= len(vectors)
            for nprobe in ann["ivf_flat"]["nprobe"] if nprobe <= nlist
        ],
        "ivf_pq": [
            {"nlist": nlist, "nprobe": nprobe, "m": ann["ivf_pq"]["m"], "nbits": ann["ivf_pq"]["nbits"]}
            for nlist in ann["ivf_pq"]["nlist"] if nlist * 39 <= len(vectors)
            for nprobe in ann["ivf_pq"]["nprobe"] if nprobe <= nlist
        ],
    }
    families = {}
    with tempfile.TemporaryDirectory(prefix="stat6207-sop-") as temporary:
        temporary = Path(temporary)
        for family, candidates in grids.items():
            grouped = {}
            for params in candidates:
                structural = tuple((key, value) for key, value in params.items() if key not in ("efSearch", "nprobe"))
                grouped.setdefault(structural, []).append(params)
            records, paths = [], {}
            for structural, variants in grouped.items():
                params = dict(structural) | variants[0]
                index = build(family, vectors.shape[1], params)
                before = process_rss_bytes()
                start = time.perf_counter()
                with ResourceMonitor(0.2) as monitor:
                    if not index.is_trained:
                        # Unsupervised index training may use the gallery. Query labels and
                        # held-out ANN results remain unavailable to parameter selection.
                        index.train(vectors)
                    index.add(vectors)
                build_record = {"seconds": time.perf_counter() - start, "rss_before_bytes": before, "rss_after_bytes": process_rss_bytes(), "resources": monitor.summary()}
                path = temporary / (family + "-" + "-".join(f"{k}{v}" for k, v in structural) + ".faiss")
                faiss.write_index(index, str(path))
                paths[structural] = path
                for candidate in variants:
                    if family == "hnsw": index.hnsw.efSearch = candidate["efSearch"]
                    else: index.nprobe = candidate["nprobe"]
                    ids, batch = timed_search(index, vectors[development], 101, 256, ann["timed_repetitions"])
                    ids = remove_self(development, ids)
                    _, single = timed_search(index, vectors[development[:1000]], 101, 1, 1)
                    name = family + "-" + "-".join(f"{key}{value}" for key, value in sorted(candidate.items()))
                    records.append({"name": name, "params": candidate, "build": build_record, "serialized_bytes": path.stat().st_size,
                                    "development": {**ann_recall(exact_ids[development, :100], ids[:, :100]), "batch_256": batch, "single": single}})
                del index
                gc.collect()
            selected = choose_candidate(records, ann["selection"]["minimum_development_recall_at_10"])
            structural = tuple((key, value) for key, value in selected["params"].items() if key not in ("efSearch", "nprobe"))
            index = faiss.read_index(str(paths[structural]))
            if family == "hnsw": index.hnsw.efSearch = selected["params"]["efSearch"]
            else: index.nprobe = selected["params"]["nprobe"]
            ids, batch = timed_search(index, vectors[held_out], 101, 256, ann["timed_repetitions"])
            ids = remove_self(held_out, ids)
            assert_no_self_match(held_out, ids)
            _, single = timed_search(index, vectors[held_out[:1000]], 101, 1, 1)
            families[family] = {"development_candidates": records, "selected": selected["name"], "selected_params": selected["params"],
                                "test": {**ann_recall(exact_ids[held_out, :100], ids[:, :100]), "batch_256": batch, "single": single}}
            # Semantic metrics for held-out queries need query-relative labels rather than a square helper.
            relevant = product_ids[ids[:, :100]] == product_ids[held_out, None]
            families[family]["test"]["semantic_recall_at_1"] = float(np.mean(np.any(relevant[:, :1], axis=1)))
            families[family]["test"]["semantic_recall_at_10"] = float(np.mean(np.any(relevant[:, :10], axis=1)))
            families[family]["test"]["semantic_recall_at_100"] = float(np.mean(np.any(relevant[:, :100], axis=1)))
            del index
            gc.collect()

    result = {
        "status": "passed",
        "config_sha256": file_hash(config_path),
        "ann_config": ann,
        "ann_config_sha256": canonical_hash(ann),
        "sop_config": sop,
        "sop_config_sha256": canonical_hash(sop),
        "sop_lock_sha256": file_hash(lock_path),
        "embedding_sha256": file_hash(vectors_path),
        "protocol": sop["protocol"],
        "gallery_images": len(vectors),
        "products": int(len(np.unique(product_ids))),
        "exact_build": exact_build,
        "exact_timing": exact_timing,
        "standard_semantic": semantic,
        "ann_development_queries": len(development),
        "ann_held_out_queries": len(held_out),
        "families": families,
    }
    (output / "retrieval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "standard_semantic": semantic, "selected": {k: v["selected_params"] for k, v in families.items()}}, indent=2))


if __name__ == "__main__":
    main()
