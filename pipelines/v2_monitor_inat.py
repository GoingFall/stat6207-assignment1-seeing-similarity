"""Build validation-only drift thresholds and evaluate locked iNaturalist splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pipelines.prepare_data import ROOT
from src.monitoring.drift import drift_metrics, estimate_rbf_gamma


EMBEDDINGS = ROOT / "data/v2.0/embeddings/inat_birds"
OUTPUT = ROOT / "results/v2.0/inat/monitoring.json"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config_path = ROOT / "configs/v2.0/experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))["monitoring"]
    encoding_path = ROOT / "results/v2.0/inat/encoding.json"
    if json.loads(encoding_path.read_text(encoding="utf-8"))["status"] != "passed":
        raise RuntimeError("Passed locked embeddings are required")
    reference = np.load(EMBEDDINGS / f"{config['reference_split']}_embeddings.npy")
    rng = np.random.default_rng(config["seed"])
    projections = rng.standard_normal((config["random_projections"], reference.shape[1])).astype(np.float32)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    rbf_gamma = estimate_rbf_gamma(reference, seed=config["seed"])
    null_records = []
    half = len(reference) // 2
    for index in range(config["bootstrap_partitions"]):
        permutation = rng.permutation(len(reference))
        left = reference[permutation[:half]]
        right = reference[permutation[half : 2 * half]]
        null_records.append(drift_metrics(
            left, right, projections, config["psi_bins"], config["rbf_mmd_samples"],
            config["seed"] + index, rbf_gamma=rbf_gamma,
        ))
    monitored = ("centroid_cosine_shift", "projection_psi", "rbf_mmd")
    thresholds = {metric: float(np.quantile([row[metric] for row in null_records], config["alert_quantile"])) for metric in monitored}
    evaluations = {}
    for split in config["evaluation_splits"]:
        current = np.load(EMBEDDINGS / f"{split}_embeddings.npy")
        metrics = drift_metrics(
            reference, current, projections, config["psi_bins"], config["rbf_mmd_samples"],
            config["seed"], rbf_gamma=rbf_gamma,
        )
        evaluations[split] = {**metrics, "alerts": {metric: metrics[metric] > thresholds[metric] for metric in monitored}}
    result = {
        "status": "passed", "config": config, "encoding_sha256": hash_file(encoding_path),
        "threshold_source": "repeated disjoint half-splits of known validation only",
        "rbf_gamma": rbf_gamma,
        "rbf_gamma_source": "known validation only; fixed for every null and evaluation comparison",
        "null_records": null_records,
        "thresholds": thresholds, "evaluations": evaluations,
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "thresholds": thresholds, "evaluations": evaluations}, indent=2))


if __name__ == "__main__":
    main()
