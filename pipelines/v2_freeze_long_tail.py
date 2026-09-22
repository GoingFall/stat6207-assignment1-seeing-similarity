"""Freeze controlled long-tail sampled image IDs before training or model selection."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pipelines.prepare_data import ROOT
from src.training.long_tail import sample_ids


OUTPUT = ROOT / "data_v2/manifests/long_tail"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    config_path = ROOT / "configs/v2/experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = ROOT / "data_v2/manifests/inat_birds_train.jsonl"
    candidates = defaultdict(list)
    for line in source_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        candidates[int(row["class_id"])].append(int(row["image_id"]))
    if len(candidates) != 1000 or {len(values) for values in candidates.values()} != {30}:
        raise AssertionError("Expected 1,000 known classes with 30 frozen training candidates each")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": 1,
        "source_manifest": source_path.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "methods": config["long_tail"]["methods"],
        "runs": [],
        "blocked": [],
    }
    for ratio in config["long_tail"]["ratios"]:
        for seed in config["long_tail"]["seeds"]:
            try:
                run = sample_ids(candidates, ratio=ratio, seed=seed)
            except ValueError as error:
                index["blocked"].append({"ratio": ratio, "seed": seed, "reason": str(error)})
                continue
            path = OUTPUT / f"ratio-{ratio}-seed-{seed}.json"
            if path.exists():
                raise RuntimeError(f"Long-tail lock exists; refusing to overwrite: {path}")
            path.write_text(json.dumps(run, indent=2), encoding="utf-8")
            index["runs"].append({
                "ratio": ratio,
                "seed": seed,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(canonical(run)).hexdigest(),
                "selected_images": sum(run["counts"].values()),
            })
    if len(index["runs"]) != 5 or {run["ratio"] for run in index["runs"]} != {10}:
        raise AssertionError("Expected exactly five feasible 10:1 frozen runs")
    if len(index["blocked"]) != 5 or {run["ratio"] for run in index["blocked"]} != {50}:
        raise AssertionError("Expected every 50:1 run to be explicitly blocked")
    index_path = OUTPUT / "index.json"
    if index_path.exists():
        raise RuntimeError(f"Long-tail index exists; refusing to overwrite: {index_path}")
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    index_path.with_suffix(".sha256").write_text(
        hashlib.sha256(canonical(index)).hexdigest() + "  index.json\n", encoding="utf-8"
    )
    print(json.dumps({"status": "locked", "feasible_runs": len(index["runs"]), "blocked_runs": len(index["blocked"])}, indent=2))


if __name__ == "__main__":
    main()
