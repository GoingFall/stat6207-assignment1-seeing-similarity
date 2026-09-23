"""Run the pre-registered DINOv2 CUDA sanity benchmark."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from pipelines.prepare_data import ROOT
from src.benchmarking.resources import ResourceMonitor


def main() -> None:
    config = json.loads((ROOT / "configs/v2.0/experiment.json").read_text())
    gate = config["gpu_gate"]
    paths = sorted((ROOT / "data/v1.0/images").glob("*.png"))[: gate["images"]]
    if len(paths) != gate["images"]:
        raise RuntimeError(f"Expected {gate['images']} local gate images, found {len(paths)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to report a GPU benchmark")

    processor = AutoImageProcessor.from_pretrained(gate["model"])
    model = AutoModel.from_pretrained(gate["model"]).to("cuda").eval()
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    warmup = processor(images=images[: gate["batch_size"]], return_tensors="pt")
    warmup = {key: value.to("cuda") for key, value in warmup.items()}
    with torch.inference_mode():
        model(**warmup)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    outputs_on_cuda = True
    processed = 0
    passes = 0
    start = time.perf_counter()
    with ResourceMonitor(gate["utilization_sample_seconds"]) as monitor:
        with torch.inference_mode():
            while time.perf_counter() - start < gate["minimum_measurement_seconds"]:
                for begin in range(0, len(images), gate["batch_size"]):
                    batch = processor(images=images[begin : begin + gate["batch_size"]], return_tensors="pt")
                    batch = {key: value.to("cuda") for key, value in batch.items()}
                    output = model(**batch).last_hidden_state[:, 0]
                    outputs_on_cuda = outputs_on_cuda and output.is_cuda
                    processed += len(output)
                passes += 1
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    resources = monitor.summary()
    active_util = resources["gpu_mean_active_utilization_percent"]
    passed = outputs_on_cuda and active_util >= gate["target_active_utilization_percent"]
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = {
        "status": "passed" if passed else "failed",
        "criterion": "mean utilization among active nvidia-smi samples",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": driver,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "model": gate["model"],
        "unique_images": len(images),
        "processed_images": processed,
        "passes": passes,
        "batch_size": gate["batch_size"],
        "elapsed_seconds": elapsed,
        "images_per_second": processed / elapsed,
        "outputs_on_cuda": outputs_on_cuda,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resources": resources,
        "target_active_utilization_percent": gate["target_active_utilization_percent"],
    }
    folder = ROOT / "results/v2.0/gates"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gpu_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError("GPU gate failed; do not start large image encoding")


if __name__ == "__main__":
    main()
