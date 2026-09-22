"""Freeze iNaturalist Birds class and image splits before training or OOD tuning."""
from __future__ import annotations

import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from pipelines.prepare_data import ROOT


SOURCE = ROOT / "data_v2/raw/inat_birds_metadata"
LOCK = ROOT / "data_v2/manifests/inat_birds_lock.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def load_maybe_zipped_json(path: Path) -> object:
    payload = path.read_bytes()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise RuntimeError(f"Expected one JSON member in {path}, found {names}")
            payload = archive.read(names[0])
    return json.loads(payload)


def table_rows(columnar: dict) -> list[dict]:
    keys = [key for key in columnar if key not in ("level_0", "index")]
    row_ids = sorted(columnar[keys[0]], key=int)
    return [{key: columnar[key][row_id] for key in keys} for row_id in row_ids]


def main() -> None:
    if LOCK.exists():
        raise RuntimeError(f"iNaturalist split is already locked at {LOCK}; refusing to overwrite")
    config_path = ROOT / "configs/v2/experiment.json"
    full_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = full_config["inat_birds"]
    classes = table_rows(load_maybe_zipped_json(SOURCE / "bird_classes.json"))
    annotations = table_rows(load_maybe_zipped_json(SOURCE / "bird_annotations.json"))
    images = table_rows(load_maybe_zipped_json(SOURCE / "bird_images.json"))
    if len(classes) != 1486 or len(annotations) != 74300 or len(images) != 74300:
        raise AssertionError(f"Unexpected metadata sizes: {len(classes)}, {len(annotations)}, {len(images)}")

    class_by_id = {int(row["id"]): row for row in classes}
    annotation_by_image = {int(row["image_id"]): int(row["category_id"]) for row in annotations}
    if len(annotation_by_image) != len(images):
        raise AssertionError("Every image must have exactly one species annotation")
    counts = Counter(annotation_by_image.values())
    if set(counts) != set(class_by_id) or set(counts.values()) != {50}:
        raise AssertionError("Expected exactly 50 images for each of 1,486 bird species")

    # Deterministic species-level split, stratified round-robin by family to spread taxonomy.
    by_family = defaultdict(list)
    for class_id, row in class_by_id.items():
        by_family[row["family"]].append(class_id)
    rng = random.Random(config["class_split_seed"])
    ordered = []
    families = sorted(by_family)
    for family in families:
        values = sorted(by_family[family])
        rng.shuffle(values)
        by_family[family] = values
    while any(by_family.values()):
        for family in families:
            if by_family[family]:
                ordered.append(by_family[family].pop())
    expected_split = config["class_split"]
    class_splits = {
        "known": sorted(ordered[: expected_split["known"]]),
        "unknown_development": sorted(ordered[expected_split["known"] : expected_split["known"] + expected_split["unknown_development"]]),
        "unknown_test": sorted(ordered[-expected_split["unknown_test"] :]),
    }
    if len(set().union(*map(set, class_splits.values()))) != len(classes):
        raise AssertionError("Class-level OOD splits overlap or omit species")

    images_by_class = defaultdict(list)
    for row in images:
        image_id = int(row["id"])
        class_id = annotation_by_image[image_id]
        images_by_class[class_id].append({
            "image_id": image_id,
            "class_id": class_id,
            "file_name": row["file_name"],
            "license": row.get("license"),
        })
    image_splits = {"train": [], "validation": [], "test": [], "unknown_development": [], "unknown_test": []}
    allocation = config["balanced_image_split_per_known_class"]
    for class_id in class_splits["known"]:
        rows = sorted(images_by_class[class_id], key=lambda row: row["image_id"])
        random.Random(config["class_split_seed"] + class_id).shuffle(rows)
        first = allocation["train"]
        second = first + allocation["validation"]
        image_splits["train"].extend(rows[:first])
        image_splits["validation"].extend(rows[first:second])
        image_splits["test"].extend(rows[second:])
    for split in ("unknown_development", "unknown_test"):
        for class_id in class_splits[split]:
            image_splits[split].extend(sorted(images_by_class[class_id], key=lambda row: row["image_id"]))

    manifest_records = {}
    for split, rows in image_splits.items():
        path = ROOT / f"data_v2/manifests/inat_birds_{split}.jsonl"
        payload = b"".join(canonical(row) + b"\n" for row in rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest_records[split] = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(payload), "images": len(rows), "classes": len(set(row["class_id"] for row in rows))}

    class_records = {
        split: [
            {key: class_by_id[class_id].get(key) for key in ("id", "name", "common_name", "order", "family", "genus", "image_dir_name")}
            for class_id in ids
        ]
        for split, ids in class_splits.items()
    }
    lock = {
        "schema_version": 1,
        "source": config["source"],
        "official_terms": config["official_terms"],
        "publish_images": False,
        "config_sha256": digest(config_path.read_bytes()),
        "source_files": {path.name: digest(path.read_bytes()) for path in sorted(SOURCE.glob("*.json"))},
        "class_split_method": "seeded family-stratified round-robin before any model inference",
        "class_splits": class_records,
        "manifests": manifest_records,
        "images_per_species": 50,
        "controlled_long_tail_ratios": full_config["long_tail"]["ratios"],
        "long_tail_50_to_1_status": "blocked: exact integer ratio is incompatible with independent 30/10/10 split",
    }
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    LOCK.with_suffix(".sha256").write_text(
        digest(canonical(lock)) + "  inat_birds_lock.json\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "locked", "classes": {k: len(v) for k, v in class_records.items()}, "manifests": manifest_records}, indent=2))


if __name__ == "__main__":
    main()
