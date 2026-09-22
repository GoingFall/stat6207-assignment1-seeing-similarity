"""Audit SIFT1M official ground-truth ties against two independent exact paths."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from pipelines.prepare_data import ROOT
from src.benchmarking.ann import recall_metrics, sharded_exact_l2
from src.benchmarking.vector_io import read_fvecs, read_ivecs


def main() -> None:
    config = json.loads((ROOT / "configs/v2/experiment.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "data_v2/manifests/sift1m_lock.json").read_text(encoding="utf-8"))
    base = read_fvecs(ROOT / lock["files"]["base"]["path"])
    count = config["exact"]["feasibility_queries"]
    queries = read_fvecs(ROOT / lock["files"]["query"]["path"])[:count]
    official = read_ivecs(ROOT / lock["files"]["groundtruth"]["path"])[:count]
    shard_distances, shard_ids = sharded_exact_l2(
        base, queries, config["exact"]["k"], config["exact"]["query_batch"], config["exact"]["database_shard"]
    )
    flat = faiss.IndexFlatL2(base.shape[1])
    flat.add(base)
    flat_distances, flat_ids = flat.search(queries, config["exact"]["k"])
    if recall_metrics(flat_ids, shard_ids)["recall_at_10"] != 1.0:
        raise AssertionError("Independent exact implementations disagree within top 10")

    mismatches = []
    for row in np.flatnonzero(official[:, 0] != shard_ids[:, 0]):
        official_id = int(official[row, 0])
        computed_id = int(shard_ids[row, 0])
        official_distance = float(np.sum((queries[row] - base[official_id]) ** 2))
        computed_distance = float(shard_distances[row, 0])
        mismatches.append({
            "query_row": int(row),
            "official_id": official_id,
            "computed_id": computed_id,
            "official_squared_l2": official_distance,
            "computed_squared_l2": computed_distance,
            "equal_distance_tie": official_distance == computed_distance,
        })
    if not mismatches or not all(record["equal_distance_tie"] for record in mismatches):
        raise AssertionError(f"Official top-1 differences are not fully explained by ties: {mismatches}")
    result = {
        "status": "passed",
        "queries": count,
        "sharded_exact_vs_faiss_flat": recall_metrics(flat_ids, shard_ids),
        "sharded_exact_vs_official": recall_metrics(official, shard_ids),
        "faiss_flat_vs_official": recall_metrics(official, flat_ids),
        "official_top1_tie_order_differences": mismatches,
        "interpretation": "Both exact implementations agree at top-1/top-10; official overlap below 1.0 is caused by equal-distance tie ordering.",
    }
    output = ROOT / "results_v2/checks/sift1m_exact_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
