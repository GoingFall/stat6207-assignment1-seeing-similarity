"""Fail-closed preflight for private-data/full-replay environments.

This check never downloads, fabricates, or modifies locked experiment inputs. It
only reports whether the local machine has the files required for a full replay
and verifies the public lock metadata when those files are available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.paths import ROOT, resolve
from src.training.provenance import sha256


def _artifact_status(path: Path, expected: str | None = None) -> dict:
    """Report a local artifact's presence and compare its digest when requested."""
    result = {
        "path": path.relative_to(ROOT).as_posix(),
        "exists": path.is_file() or path.is_dir(),
        "kind": "directory" if path.is_dir() else "file",
        "hash_matches": None,
    }
    if path.is_file():
        result["bytes"] = path.stat().st_size
        result["sha256"] = sha256(path)
        if expected is not None:
            result["hash_matches"] = result["sha256"] == expected
    return result


def main() -> None:
    """Report local replay prerequisites and fail when required evidence is absent.

    Returns:
        None. Emits the evidence summary as JSON on stdout.
    """
    checks = []
    required = [
        ROOT / "data/v2.0/manifests/inat_birds_images.jsonl",
        ROOT / "data/v2.0/manifests/inat_birds_images_lock.json",
        ROOT / "data/v2.1/manifests/lock.json",
        ROOT / "data/v2.1/embeddings/inat_birds",
    ]
    for path in required:
        checks.append(_artifact_status(path))

    lock_path = ROOT / "data/v2.1/manifests/lock.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        # A lock without the config digest cannot support a fail-closed preflight, so treat the
        # missing field as a failure rather than reporting readiness on an unverified config.
        config = resolve("configs/v2.1/experiment.json")
        checks.append(_artifact_status(config, lock["config_sha256"]))

    source_manifest = ROOT / "data/v2.0/manifests/inat_birds_images.jsonl"
    missing_source_images = None
    if source_manifest.is_file():
        rows = [
            json.loads(line)
            for line in source_manifest.read_text(encoding="utf-8").splitlines()
        ]
        missing_source_images = sum(
            not resolve(row["local_path"]).is_file() for row in rows
        )
    mismatches = [item["path"] for item in checks if item["hash_matches"] is False]
    hashes_match = not mismatches
    complete = (
        all(item["exists"] for item in checks)
        and hashes_match
        and missing_source_images == 0
    )
    if complete:
        status = "ready"
    elif not hashes_match:
        status = "verification-failed"
    else:
        status = "local-only-verification"
    payload = {
        "status": status,
        "full_replay_requires": [
            "private source image manifest and licensed source images",
            "hash-locked V2 embeddings",
            "original experiment environment and model dependencies",
        ],
        "checks": checks,
        "hash_mismatches": mismatches,
        "source_image_count_missing": missing_source_images,
        "note": "Missing private inputs are a reproducibility boundary; this preflight does not create substitutes.",
    }
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
