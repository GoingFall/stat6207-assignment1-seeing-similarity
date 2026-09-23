"""Small HTTP health/status service; retrieval serving is added only after release locking."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def status_payload(root: Path = ROOT) -> dict:
    artifacts = {
        "gpu_gate": root / "results/v2.0/gates/gpu_gate.json",
        "faiss_smoke": root / "results/v2.0/gates/faiss_smoke.json",
        "sop_lock": root / "data/v2.0/manifests/sop_lock.json",
        "sop_retrieval": root / "results/v2.0/sop/retrieval.json",
        "sift1m_lock": root / "data/v2.0/manifests/sift1m_lock.json",
        "sift1m_benchmark": root / "results/v2.0/ann/sift1m.json",
        "sift1m_scale_sweep": root / "results/v2.0/ann/sift1m_scales.json",
        "inat_image_lock": root / "data/v2.0/manifests/inat_birds_images_lock.json",
        "inat_encoding": root / "results/v2.0/inat/encoding.json",
        "inat_training": root / "results/v2.0/inat/training.json",
        "inat_open_set": root / "results/v2.0/inat/open_set.json",
        "inat_monitoring": root / "results/v2.0/inat/monitoring.json",
    }
    return {
        "service": "stat6207-retrieval-v2",
        "status": "healthy",
        "version": (root / "VERSION").read_text(encoding="utf-8").strip(),
        "artifacts": {name: path.exists() for name, path in artifacts.items()},
        "revision_2_1": {
            name: (root / path).exists()
            for name, path in {
                "split_lock": "data/v2.1/manifests/lock.json",
                "training": "results/v2.1/inat/training.json",
                "open_set": "results/v2.1/inat/open_set.json",
                "monitoring": "results/v2.1/inat/monitoring.json",
                "verification": "results/v2.1/verification.json",
            }.items()
        },
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/health":
            payload, code = {"status": "healthy"}, 200
        elif self.path == "/v1/status":
            payload, code = status_payload(), 200
        else:
            payload, code = {"error": "not_found"}, 404
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"status": "listening", "host": args.host, "port": server.server_port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
