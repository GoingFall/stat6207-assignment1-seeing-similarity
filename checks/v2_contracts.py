"""Version 2.0 split, self-match, manifest and ANN correctness contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_self_match(query_ids: np.ndarray, result_ids: np.ndarray) -> None:
    if len(query_ids) != len(result_ids):
        raise AssertionError("Query and result row counts differ")
    collisions = np.flatnonzero(query_ids == result_ids[:, 0])
    if len(collisions):
        raise AssertionError(f"Top-1 self matches at rows {collisions[:10].tolist()}")


def assert_product_disjoint(train_product_ids: set[int], test_product_ids: set[int]) -> None:
    overlap = train_product_ids & test_product_ids
    if overlap:
        raise AssertionError(f"Product split overlap: {sorted(overlap)[:10]}")


def verify_hash_manifest(manifest_path: Path, root: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise AssertionError(f"Hash mismatch for {relative}: {actual} != {expected}")
