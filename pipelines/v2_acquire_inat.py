"""Acquire local iNaturalist Birds images and freeze file hashes before inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import kagglehub

from pipelines.prepare_data import ROOT


DATASET = "sharansmenon/inat2021birds/versions/1"
RAW = ROOT / "data_v2/raw/inat_birds"
METADATA_LOCK = ROOT / "data_v2/manifests/inat_birds_lock.json"
IMAGE_LOCK = ROOT / "data_v2/manifests/inat_birds_images_lock.json"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    if IMAGE_LOCK.exists():
        raise RuntimeError(f"iNaturalist image lock already exists at {IMAGE_LOCK}; refusing to overwrite")
    metadata_lock = json.loads(METADATA_LOCK.read_text(encoding="utf-8"))
    if metadata_lock["source"] != DATASET or metadata_lock["publish_images"] is not False:
        raise AssertionError("Metadata lock source/publication policy mismatch")
    downloaded = Path(kagglehub.dataset_download(DATASET, output_dir=str(RAW)))
    by_name = {}
    duplicates = set()
    for path in downloaded.rglob("*.jpg"):
        if path.name in by_name:
            duplicates.add(path.name)
        else:
            by_name[path.name] = path

    rows = []
    seen_ids = set()
    for split, record in metadata_lock["manifests"].items():
        for row in read_manifest(ROOT / record["path"]):
            image_id = int(row["image_id"])
            if image_id in seen_ids:
                raise AssertionError(f"Image ID appears in multiple frozen splits: {image_id}")
            seen_ids.add(image_id)
            expected = Path(row["file_name"])
            candidates = [downloaded / expected, RAW / expected]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                if expected.name in duplicates or expected.name not in by_name:
                    raise FileNotFoundError(f"Cannot uniquely locate frozen image {expected}")
                path = by_name[expected.name]
            try:
                local_path = path.relative_to(ROOT).as_posix()
            except ValueError as error:
                raise RuntimeError(f"Downloaded image is outside the Version 2.0 workspace: {path}") from error
            rows.append({
                "image_id": image_id,
                "class_id": int(row["class_id"]),
                "split": split,
                "source_file_name": row["file_name"],
                "local_path": local_path,
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
                "license": row.get("license"),
            })
    if len(rows) != 74300 or len(seen_ids) != 74300:
        raise AssertionError(f"Expected 74,300 unique frozen images, found {len(rows)}/{len(seen_ids)}")
    rows.sort(key=lambda row: row["image_id"])
    image_manifest = ROOT / "data_v2/manifests/inat_birds_images.jsonl"
    payload = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)
    image_manifest.write_bytes(payload)
    lock = {
        "schema_version": 1,
        "source": DATASET,
        "metadata_lock_sha256": hash_file(METADATA_LOCK),
        "images": len(rows),
        "manifest": image_manifest.relative_to(ROOT).as_posix(),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "total_bytes": sum(row["bytes"] for row in rows),
        "publish_images": False,
    }
    IMAGE_LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    IMAGE_LOCK.with_suffix(".sha256").write_text(
        canonical_hash(lock) + "  inat_birds_images_lock.json\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
