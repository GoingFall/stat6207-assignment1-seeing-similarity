"""Verify, safely extract and lock the official CC0 SIFT1M benchmark."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from pipelines.prepare_data import ROOT
from src.benchmarking.vector_io import read_fvecs, read_ivecs


RAW = ROOT / "data/v2.0/raw/sift1m"
ARCHIVE = RAW / "sift.tar.gz"
OUTPUT = ROOT / "data/v2.0/processed/sift1m"
LOCK = ROOT / "data/v2.0/manifests/sift1m_lock.json"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    if LOCK.exists():
        raise RuntimeError(f"SIFT1M is already locked at {LOCK}; refusing to overwrite")
    config = json.loads((ROOT / "configs/v2.0/experiment.json").read_text())["sift1m"]
    if file_hash(ARCHIVE, "md5") != config["archive_md5"]:
        raise RuntimeError("Official SIFT1M archive MD5 mismatch")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        for member in members:
            target = (OUTPUT / member.name).resolve()
            if OUTPUT.resolve() not in target.parents:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
        archive.extractall(OUTPUT, members=members, filter="data")

    candidates = {
        "base": next(OUTPUT.rglob("sift_base.fvecs")),
        "query": next(OUTPUT.rglob("sift_query.fvecs")),
        "learn": next(OUTPUT.rglob("sift_learn.fvecs")),
        "groundtruth": next(OUTPUT.rglob("sift_groundtruth.ivecs")),
    }
    arrays = {
        "base": read_fvecs(candidates["base"], 128),
        "query": read_fvecs(candidates["query"], 128),
        "learn": read_fvecs(candidates["learn"], 128),
        "groundtruth": read_ivecs(candidates["groundtruth"], 100),
    }
    for name, expected in config["expected"].items():
        if list(arrays[name].shape) != expected:
            raise AssertionError(f"{name}: {arrays[name].shape} != {expected}")
    if arrays["groundtruth"].min() < 0 or arrays["groundtruth"].max() >= len(arrays["base"]):
        raise AssertionError("Ground-truth indices are outside the base-vector range")
    files = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
            "shape": list(arrays[name].shape),
        }
        for name, path in candidates.items()
    }
    lock = {
        "schema_version": 1,
        "source": config["source"],
        "official_page": config["official_page"],
        "license": config["license"],
        "archive": {"bytes": ARCHIVE.stat().st_size, "md5": config["archive_md5"], "sha256": file_hash(ARCHIVE)},
        "files": files,
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    LOCK.with_suffix(".sha256").write_text(
        canonical_hash(lock) + "  sift1m_lock.json\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
