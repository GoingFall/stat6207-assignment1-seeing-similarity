"""Package publishable Version 2.0 artifacts while excluding all local-only datasets/models."""
from __future__ import annotations

import hashlib
import json
import zipfile
import tempfile
from pathlib import Path

from pipelines.prepare_data import ROOT


OUTPUT = ROOT / "releases/2.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if (ROOT / "VERSION").read_text(encoding="utf-8").strip() != "2.0":
        raise RuntimeError("Set VERSION to 2.0 only after every Version 2.0 release gate passes")
    verification_path = ROOT / "results_v2/checks/verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    required = {
        "sop_retrieval", "sift1m_benchmark", "sift1m_scale_sweep", "inat_image_lock", "inat_embeddings",
        "inat_training", "inat_open_set", "inat_monitoring",
    }
    if verification["status"] != "passed" or not required.issubset(verification["checks"]):
        raise RuntimeError(f"Version 2.0 verification is incomplete: missing {sorted(required - set(verification['checks']))}")

    files = []
    for name in ("README.md", "VERSION", "requirements.txt", "environment.yml"):
        path = ROOT / name
        if path.exists():
            files.append(path)
    release_notes = ROOT / "releases/2.0/RELEASE-2.0.md"
    if release_notes.exists():
        files.append(release_notes)
    for directory in ("src", "pipelines", "checks", "reporting", "tools", "configs/v2", ".github/workflows"):
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts)
    files.extend(path for path in (ROOT / "docs").rglob("*.md") if path.is_file())
    private_inat_files = {
        "inat_birds_lock.json",
        "inat_birds_lock.sha256",
        "inat_birds_train.jsonl",
        "inat_birds_validation.jsonl",
        "inat_birds_test.jsonl",
        "inat_birds_unknown_development.jsonl",
        "inat_birds_unknown_test.jsonl",
        "inat_birds_images.jsonl",
        "inat_birds_images_lock.json",
        "inat_birds_images_lock.sha256",
    }
    files.extend(
        path for path in (ROOT / "data_v2/manifests").rglob("*")
        if path.is_file() and path.name not in private_inat_files
    )
    files.extend(path for path in (ROOT / "results_v2").rglob("*.json") if path.is_file() and not path.name.endswith("partial.json"))
    files.extend(path for path in (ROOT / "results_v2/figures").glob("*.png") if path.is_file())
    forbidden_parts = {"raw", "processed", "embeddings"}
    forbidden_suffixes = {".npy", ".npz", ".pt", ".faiss"}
    selected = []
    for path in sorted(set(files), key=lambda value: value.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT)
        if forbidden_parts.intersection(relative.parts) or path.suffix.lower() in forbidden_suffixes:
            continue
        if relative.as_posix().startswith("releases/1.0/") or path.name == "retrieval-superseded-undertrained-ivf.json":
            continue
        selected.append(path)
    manifest = {
        "version": "2.0",
        "publication_boundary": "code, configs, documentation, hashes/manifests, aggregate metrics and figures only; no source images, embeddings, models, indexes or per-query neighbours",
        "files": {
            path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in selected
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "MANIFEST-2.0.json"
    archive_path = OUTPUT / "Assignment1-2.0.zip"
    if manifest_path.exists() or archive_path.exists():
        raise RuntimeError("Version 2.0 release artifacts already exist; refusing to overwrite")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    image_manifest = ROOT / "data_v2/manifests/inat_birds_images.jsonl"
    with tempfile.TemporaryDirectory(prefix="stat6207-release-") as folder:
        generated = {}
        private_metadata_lock = json.loads((ROOT / "data_v2/manifests/inat_birds_lock.json").read_text(encoding="utf-8"))
        public_metadata_records = {}
        for split, record in private_metadata_lock["manifests"].items():
            public_path = Path(folder) / f"inat_birds_{split}_public.jsonl"
            with (ROOT / record["path"]).open(encoding="utf-8") as source, public_path.open("w", encoding="utf-8", newline="\n") as target:
                for line in source:
                    row = json.loads(line)
                    public = {key: row[key] for key in ("image_id", "class_id", "license")}
                    target.write(json.dumps(public, sort_keys=True, separators=(",", ":")) + "\n")
            archive_name = f"data_v2/manifests/inat_birds_{split}_public.jsonl"
            public_metadata_records[split] = {
                **record,
                "path": archive_name,
                "sha256": sha256(public_path),
            }
            generated[archive_name] = public_path
        public_metadata_lock = {
            **private_metadata_lock,
            "schema_version": 2,
            "class_splits": {
                split: [{key: value for key, value in row.items() if key != "image_dir_name"} for row in rows]
                for split, rows in private_metadata_lock["class_splits"].items()
            },
            "manifests": public_metadata_records,
            "publication_redaction": "source path fields and image directory names removed",
        }
        public_metadata_lock_path = Path(folder) / "inat_birds_public_lock.json"
        public_metadata_lock_path.write_text(json.dumps(public_metadata_lock, indent=2), encoding="utf-8")
        public_metadata_hash = hashlib.sha256(
            json.dumps(public_metadata_lock, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        public_metadata_sidecar = Path(folder) / "inat_birds_public_lock.sha256"
        public_metadata_sidecar.write_text(public_metadata_hash + "  inat_birds_public_lock.json\n", encoding="utf-8")
        generated["data_v2/manifests/inat_birds_public_lock.json"] = public_metadata_lock_path
        generated["data_v2/manifests/inat_birds_public_lock.sha256"] = public_metadata_sidecar

        redacted_path = Path(folder) / "inat_birds_images_public.jsonl"
        with image_manifest.open(encoding="utf-8") as source, redacted_path.open("w", encoding="utf-8", newline="\n") as target:
            for line in source:
                row = json.loads(line)
                public = {key: row[key] for key in ("image_id", "class_id", "split", "bytes", "sha256", "license")}
                target.write(json.dumps(public, sort_keys=True, separators=(",", ":")) + "\n")
        private_lock = json.loads((ROOT / "data_v2/manifests/inat_birds_images_lock.json").read_text(encoding="utf-8"))
        public_lock = {
            **private_lock,
            "schema_version": 2,
            "metadata_lock_sha256": sha256(public_metadata_lock_path),
            "manifest": "data_v2/manifests/inat_birds_images_public.jsonl",
            "manifest_sha256": sha256(redacted_path),
            "manifest_redaction": "source path fields removed; image IDs, class IDs, splits, byte sizes, hashes and licenses retained",
        }
        public_lock_path = Path(folder) / "inat_birds_images_public_lock.json"
        public_lock_path.write_text(json.dumps(public_lock, indent=2), encoding="utf-8")
        public_lock_hash = hashlib.sha256(
            json.dumps(public_lock, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        public_lock_sidecar = Path(folder) / "inat_birds_images_public_lock.sha256"
        public_lock_sidecar.write_text(public_lock_hash + "  inat_birds_images_public_lock.json\n", encoding="utf-8")
        generated["data_v2/manifests/inat_birds_images_public.jsonl"] = redacted_path
        generated["data_v2/manifests/inat_birds_images_public_lock.json"] = public_lock_path
        generated["data_v2/manifests/inat_birds_images_public_lock.sha256"] = public_lock_sidecar
        manifest["generated_public_files"] = {
            name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in generated.items()
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in selected:
                archive.write(path, path.relative_to(ROOT))
            for name, path in generated.items():
                archive.write(path, name)
            archive.write(manifest_path, manifest_path.name)
    archive_hash = sha256(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(archive_hash + "  Assignment1-2.0.zip\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "files": len(selected), "archive": archive_path.relative_to(ROOT).as_posix(), "sha256": archive_hash}, indent=2))


if __name__ == "__main__":
    main()
