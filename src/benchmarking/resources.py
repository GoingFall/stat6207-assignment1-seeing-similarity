"""Process and GPU resource sampling with JSON-serializable results."""
from __future__ import annotations

import csv
import io
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil


def process_rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def gpu_snapshot() -> dict | None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
        row = next(csv.reader(io.StringIO(output)))
        return {
            "utilization_percent": float(row[0].strip()),
            "memory_used_mib": float(row[1].strip()),
            "memory_total_mib": float(row[2].strip()),
            "power_watts": float(row[3].strip()),
        }
    except (FileNotFoundError, subprocess.SubprocessError, StopIteration, ValueError):
        return None


@dataclass
class ResourceMonitor:
    interval_seconds: float = 0.5
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    samples: list[dict] = field(default_factory=list, init=False)
    peak_rss_bytes: int = field(default=0, init=False)

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = process_rss_bytes()
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.samples.append({"time": time.time(), "rss_bytes": rss, "gpu": gpu_snapshot()})
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "ResourceMonitor":
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 3))

    def summary(self) -> dict:
        gpu = [sample["gpu"] for sample in self.samples if sample["gpu"] is not None]
        active = [sample for sample in gpu if sample["utilization_percent"] > 0]
        return {
            "samples": len(self.samples),
            "peak_rss_bytes": self.peak_rss_bytes,
            "gpu_samples": len(gpu),
            "gpu_active_samples": len(active),
            "gpu_mean_active_utilization_percent": (
                sum(sample["utilization_percent"] for sample in active) / len(active) if active else 0.0
            ),
            "gpu_peak_utilization_percent": max((sample["utilization_percent"] for sample in gpu), default=0.0),
            "gpu_peak_memory_used_mib": max((sample["memory_used_mib"] for sample in gpu), default=0.0),
        }
