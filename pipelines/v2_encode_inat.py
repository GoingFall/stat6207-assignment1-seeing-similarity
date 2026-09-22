"""Encode the fully hashed iNaturalist Birds manifests with frozen DINOv2."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from pipelines.prepare_data import ROOT
from src.benchmarking.resources import ResourceMonitor


IMAGE_LOCK = ROOT / "data_v2/manifests/inat_birds_images_lock.json"
OUTPUT = ROOT / "data_v2/embeddings/inat_birds"
RESULT = ROOT / "results_v2/inat/encoding.json"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config_path = ROOT / "configs/v2/experiment.json"
    full_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = full_config["sop"]["encoder"]
    batch_size = full_config["sop"]["batch_size"]
    lock = json.loads(IMAGE_LOCK.read_text(encoding="utf-8"))
    manifest_path = ROOT / lock["manifest"]
    if hash_file(manifest_path) != lock["manifest_sha256"]:
        raise AssertionError("iNaturalist image manifest hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required after the passed GPU gate")
    by_split = defaultdict(list)
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_split[row["split"]].append(row)
    expected = {"train": 30000, "validation": 10000, "test": 10000, "unknown_development": 12150, "unknown_test": 12150}
    if {key: len(value) for key, value in by_split.items()} != expected:
        raise AssertionError("iNaturalist image-lock split counts differ from the frozen protocol")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda").eval()
    if RESULT.exists():
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        if result.get("image_lock_sha256") != hash_file(IMAGE_LOCK) or result.get("model") != model_name:
            raise RuntimeError("Existing partial encoding does not match the current locked inputs/model")
    else:
        result = {
            "status": "running",
            "model": model_name,
            "batch_size": batch_size,
            "image_lock_sha256": hash_file(IMAGE_LOCK),
            "splits": {},
        }
    for split in expected:
        if split in result["splits"]:
            record = result["splits"][split]
            if all(hash_file(ROOT / artifact["path"]) == artifact["sha256"] for artifact in record["artifacts"].values()):
                continue
            raise RuntimeError(f"Existing {split} encoding artifacts fail hash validation")
        expected_paths = [OUTPUT / f"{split}_{suffix}.npy" for suffix in ("embeddings", "labels", "image_ids")]
        if any(path.exists() for path in expected_paths):
            raise RuntimeError(f"Untracked partial artifacts exist for {split}; refusing to overwrite")
        rows = sorted(by_split[split], key=lambda row: row["image_id"])
        chunks = []
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with ResourceMonitor(0.5) as monitor:
            with torch.inference_mode():
                for begin in range(0, len(rows), batch_size):
                    images = []
                    for row in rows[begin : begin + batch_size]:
                        path = ROOT / row["local_path"]
                        if hash_file(path) != row["sha256"]:
                            raise AssertionError(f"Image changed after lock: {row['image_id']}")
                        with Image.open(path) as image:
                            images.append(image.convert("RGB"))
                    inputs = processor(images=images, return_tensors="pt")
                    inputs = {key: value.to("cuda", non_blocking=True) for key, value in inputs.items()}
                    embedding = model(**inputs).last_hidden_state[:, 0]
                    embedding = torch.nn.functional.normalize(embedding, dim=1)
                    chunks.append(embedding.cpu().numpy().astype(np.float32))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        vectors = np.concatenate(chunks)
        labels = np.array([row["class_id"] for row in rows], dtype=np.int64)
        image_ids = np.array([row["image_id"] for row in rows], dtype=np.int64)
        if vectors.shape != (len(rows), 384) or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-6):
            raise AssertionError(f"Invalid {split} embeddings")
        paths = {
            "embeddings": OUTPUT / f"{split}_embeddings.npy",
            "labels": OUTPUT / f"{split}_labels.npy",
            "image_ids": OUTPUT / f"{split}_image_ids.npy",
        }
        np.save(paths["embeddings"], vectors, allow_pickle=False)
        np.save(paths["labels"], labels, allow_pickle=False)
        np.save(paths["image_ids"], image_ids, allow_pickle=False)
        result["splits"][split] = {
            "images": len(rows),
            "seconds": elapsed,
            "images_per_second": len(rows) / elapsed,
            "artifacts": {
                name: {"path": path.relative_to(ROOT).as_posix(), "sha256": hash_file(path)}
                for name, path in paths.items()
            },
            "resources": monitor.summary(),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["status"] = "passed"
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
