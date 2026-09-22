"""Extract SOP TFRecords and freeze official split/image manifests before encoding."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from tfrecord.reader import tfrecord_loader

from pipelines.prepare_data import ROOT


SOURCE = ROOT / "data_v2/raw/sop/stanford_online_products/1.0.0"
OUTPUT = ROOT / "data_v2/processed/sop"
LOCK = ROOT / "data_v2/manifests/sop_lock.json"


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def scalar(value) -> int:
    return int(value.item() if hasattr(value, "item") else value)


def encoded_image(record: dict) -> bytes:
    for key in ("image/encoded", "image"):
        if key in record:
            value = record[key]
            return value.item() if hasattr(value, "item") else bytes(value)
    raise KeyError(f"No encoded image feature; available keys: {sorted(record)}")


def main() -> None:
    if LOCK.exists():
        raise RuntimeError(f"SOP manifest is already locked at {LOCK}; refusing to overwrite")
    info_path = SOURCE / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Download SOP first; missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected = {split["name"]: int(split["statistics"]["numExamples"]) for split in info["splits"]}
    manifests = {}
    split_products = {}

    for split in ("train", "test"):
        shards = sorted(SOURCE.glob(f"stanford_online_products-{split}.tfrecord-*"))
        if len(shards) != 16:
            raise RuntimeError(f"Expected 16 {split} shards, found {len(shards)}")
        folder = OUTPUT / split
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        counts = Counter()
        for shard_index, shard in enumerate(shards):
            for row_index, record in enumerate(tfrecord_loader(str(shard), None)):
                class_id = scalar(record["class_id"])
                super_class_id = scalar(record["super_class_id"])
                payload = encoded_image(record)
                image_id = f"{split}-{shard_index:02d}-{row_index:05d}"
                suffix = Image.open(__import__("io").BytesIO(payload)).format.lower()
                suffix = "jpg" if suffix == "jpeg" else suffix
                relative = Path(split) / f"{image_id}.{suffix}"
                target = OUTPUT / relative
                target.write_bytes(payload)
                records.append({
                    "image_id": image_id,
                    "product_id": class_id,
                    "super_class_id": super_class_id,
                    "path": relative.as_posix(),
                    "sha256": digest(payload),
                    "bytes": len(payload),
                    "source_shard": shard.name,
                    "source_row": row_index,
                })
                counts[class_id] += 1
        if len(records) != expected[split]:
            raise AssertionError(f"{split}: extracted {len(records)}, expected {expected[split]}")
        if min(counts.values()) < 2:
            raise AssertionError(f"{split}: a product has fewer than two images")
        manifest_path = ROOT / f"data_v2/manifests/sop_{split}.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_bytes = b"".join(canonical(record) + b"\n" for record in records)
        manifest_path.write_bytes(manifest_bytes)
        manifests[split] = {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": digest(manifest_bytes),
            "images": len(records),
            "products": len(counts),
            "min_images_per_product": min(counts.values()),
            "max_images_per_product": max(counts.values()),
        }
        split_products[split] = set(counts)

    overlap = split_products["train"] & split_products["test"]
    if overlap:
        raise AssertionError(f"Official product split overlaps: {sorted(overlap)[:10]}")
    lock = {
        "schema_version": 1,
        "source": "ryanholbrook/stanford-online-products/versions/1",
        "source_dataset_info_sha256": digest(info_path.read_bytes()),
        "protocol": "official product-disjoint train/test; leave-one-out retrieval within test",
        "self_exclusion": "exact image_id only",
        "product_overlap": 0,
        "manifests": manifests,
    }
    LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    (LOCK.with_suffix(".sha256")).write_text(digest(canonical(lock)) + "  sop_lock.json\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
