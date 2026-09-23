"""Encode frozen SOP manifests with DINOv2 and record compute resources."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from pipelines.prepare_data import ROOT
from src.benchmarking.resources import ResourceMonitor
from src.paths import resolve


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    config_path = ROOT / "configs/v2.0/experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))["sop"]
    lock_path = ROOT / "data/v2.0/manifests/sop_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    output = ROOT / "data/v2.0/embeddings/sop"
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"dinov2_{split}.npy").exists() for split in ("train", "test")):
        raise RuntimeError(f"SOP embeddings already exist in {output}; refusing to overwrite")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA gate is required before SOP encoding")

    processor = AutoImageProcessor.from_pretrained(config["encoder"])
    model = AutoModel.from_pretrained(config["encoder"]).to("cuda").eval()
    records = {
        split: load_manifest(resolve(lock["manifests"][split]["path"]))
        for split in ("train", "test")
    }
    result = {
        "status": "running",
        "config_sha256": file_hash(config_path),
        "sop_config": config,
        "sop_config_sha256": canonical_hash(config),
        "sop_lock_sha256": file_hash(lock_path),
        "model": config["encoder"],
        "embedding": config["embedding"],
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "splits": {},
    }
    torch.cuda.reset_peak_memory_stats()
    for split in ("train", "test"):
        manifest = records[split]
        chunks = []
        start = time.perf_counter()
        with ResourceMonitor(0.5) as monitor:
            with torch.inference_mode():
                for begin in range(0, len(manifest), config["batch_size"]):
                    images = []
                    for row in manifest[begin : begin + config["batch_size"]]:
                        with Image.open(ROOT / "data/v2.0/processed/sop" / row["path"]) as image:
                            images.append(image.convert("RGB"))
                    inputs = processor(images=images, return_tensors="pt")
                    inputs = {key: value.to("cuda", non_blocking=True) for key, value in inputs.items()}
                    embedding = model(**inputs).last_hidden_state[:, 0]
                    embedding = torch.nn.functional.normalize(embedding, dim=1)
                    chunks.append(embedding.cpu().numpy().astype(np.float32))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        embeddings = np.concatenate(chunks)
        if embeddings.shape != (len(manifest), 384):
            raise AssertionError(f"Unexpected {split} embedding shape: {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-6):
            raise AssertionError(f"{split} embeddings are not unit normalized")
        path = output / f"dinov2_{split}.npy"
        np.save(path, embeddings, allow_pickle=False)
        ids_path = output / f"dinov2_{split}_ids.json"
        ids_path.write_text(json.dumps([row["image_id"] for row in manifest], separators=(",", ":")), encoding="utf-8")
        result["splits"][split] = {
            "images": len(manifest),
            "shape": list(embeddings.shape),
            "seconds": elapsed,
            "images_per_second": len(manifest) / elapsed,
            "embedding_path": path.relative_to(ROOT).as_posix(),
            "embedding_sha256": file_hash(path),
            "ids_path": ids_path.relative_to(ROOT).as_posix(),
            "ids_sha256": file_hash(ids_path),
            "resources": monitor.summary(),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del embeddings, chunks
    result["status"] = "passed"
    folder = ROOT / "results/v2.0/sop"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "encoding.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
