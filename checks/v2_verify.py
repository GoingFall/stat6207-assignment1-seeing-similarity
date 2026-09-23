"""Verify authoritative Version 2.0 locks and completed benchmark artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from checks.v2_contracts import assert_no_self_match, assert_product_disjoint, sha256_file
from pipelines.prepare_data import ROOT
from src.paths import resolve


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_sidecar(lock_path: Path) -> None:
    expected = lock_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = canonical_hash(json.loads(lock_path.read_text(encoding="utf-8")))
    if actual != expected:
        raise AssertionError(f"Canonical lock hash mismatch: {lock_path}")


def main() -> None:
    checks = []
    sop_lock_path = ROOT / "data/v2.0/manifests/sop_lock.json"
    sop_lock = json.loads(sop_lock_path.read_text(encoding="utf-8"))
    verify_sidecar(sop_lock_path)
    manifests = {}
    for split in ("train", "test"):
        record = sop_lock["manifests"][split]
        path = resolve(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise AssertionError(f"SOP {split} manifest hash mismatch")
        manifests[split] = jsonl(path)
        if len(manifests[split]) != record["images"]:
            raise AssertionError(f"SOP {split} manifest count mismatch")
    assert_product_disjoint(
        {row["product_id"] for row in manifests["train"]},
        {row["product_id"] for row in manifests["test"]},
    )
    checks.append("sop_lock")

    encoding_path = ROOT / "results/v2.0/sop/encoding.json"
    encoding = json.loads(encoding_path.read_text(encoding="utf-8"))
    if encoding["status"] != "passed":
        raise AssertionError("SOP encoding did not pass")
    for split in ("train", "test"):
        record = encoding["splits"][split]
        embedding_path = resolve(record["embedding_path"])
        ids_path = resolve(record["ids_path"])
        if sha256_file(embedding_path) != record["embedding_sha256"] or sha256_file(ids_path) != record["ids_sha256"]:
            raise AssertionError(f"SOP {split} embedding artifact hash mismatch")
        vectors = np.load(embedding_path, mmap_mode="r")
        if list(vectors.shape) != record["shape"] or not np.all(np.isfinite(vectors)):
            raise AssertionError(f"SOP {split} embedding shape/finiteness mismatch")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-6):
            raise AssertionError(f"SOP {split} embeddings are not unit normalized")
    checks.append("sop_embeddings")

    retrieval = json.loads((ROOT / "results/v2.0/sop/retrieval.json").read_text(encoding="utf-8"))
    if retrieval["status"] != "passed" or retrieval["gallery_images"] != len(manifests["test"]):
        raise AssertionError("Authoritative SOP retrieval result is incomplete")
    neighbours = np.load(ROOT / "results/v2.0/sop/exact_top1000.npy", mmap_mode="r")
    if neighbours.shape != (len(manifests["test"]), 1000):
        raise AssertionError("SOP exact neighbour matrix has the wrong shape")
    assert_no_self_match(np.arange(len(neighbours)), neighbours)
    checks.append("sop_retrieval")

    inat_lock_path = ROOT / "data/v2.0/manifests/inat_birds_lock.json"
    inat_lock = json.loads(inat_lock_path.read_text(encoding="utf-8"))
    verify_sidecar(inat_lock_path)
    class_sets = {
        split: {row["id"] for row in rows}
        for split, rows in inat_lock["class_splits"].items()
    }
    if sum(map(len, class_sets.values())) != len(set().union(*class_sets.values())):
        raise AssertionError("iNaturalist class-level splits overlap")
    for record in inat_lock["manifests"].values():
        if sha256_file(resolve(record["path"])) != record["sha256"]:
            raise AssertionError(f"iNaturalist manifest hash mismatch: {record['path']}")
    checks.append("inat_metadata_lock")

    inat_image_lock_path = ROOT / "data/v2.0/manifests/inat_birds_images_lock.json"
    if inat_image_lock_path.exists():
        verify_sidecar(inat_image_lock_path)
        image_lock = json.loads(inat_image_lock_path.read_text(encoding="utf-8"))
        image_manifest = resolve(image_lock["manifest"])
        if sha256_file(image_manifest) != image_lock["manifest_sha256"]:
            raise AssertionError("iNaturalist image manifest hash mismatch")
        rows = jsonl(image_manifest)
        if len(rows) != image_lock["images"] or len({row["image_id"] for row in rows}) != image_lock["images"]:
            raise AssertionError("iNaturalist image lock count or uniqueness mismatch")
        checks.append("inat_image_lock")

    long_tail_index_path = ROOT / "data/v2.0/manifests/long_tail/index.json"
    if long_tail_index_path.exists():
        verify_sidecar(long_tail_index_path)
        long_tail = json.loads(long_tail_index_path.read_text(encoding="utf-8"))
        if len(long_tail["runs"]) != 5 or len(long_tail["blocked"]) != 5:
            raise AssertionError("Long-tail run/blocker counts differ from the frozen protocol")
        for record in long_tail["runs"]:
            run_path = resolve(record["path"])
            if canonical_hash(json.loads(run_path.read_text(encoding="utf-8"))) != record["sha256"]:
                raise AssertionError(f"Long-tail lock hash mismatch: {run_path}")
        checks.append("long_tail_locks")

    sift_lock_path = ROOT / "data/v2.0/manifests/sift1m_lock.json"
    sift_result_path = ROOT / "results/v2.0/ann/sift1m.json"
    if sift_lock_path.exists():
        lock = json.loads(sift_lock_path.read_text(encoding="utf-8"))
        verify_sidecar(sift_lock_path)
        for record in lock["files"].values():
            if sha256_file(resolve(record["path"])) != record["sha256"]:
                raise AssertionError(f"SIFT1M file hash mismatch: {record['path']}")
        checks.append("sift1m_lock")
    if sift_result_path.exists():
        if json.loads(sift_result_path.read_text(encoding="utf-8"))["status"] != "passed":
            raise AssertionError("SIFT1M benchmark did not pass")
        audit_path = ROOT / "results/v2.0/checks/sift1m_exact_audit.json"
        if not audit_path.exists() or json.loads(audit_path.read_text(encoding="utf-8"))["status"] != "passed":
            raise AssertionError("SIFT1M exact tie audit is missing or failed")
        checks.append("sift1m_benchmark")
    sift_scales_path = ROOT / "results/v2.0/ann/sift1m_scales.json"
    if sift_scales_path.exists():
        scales = json.loads(sift_scales_path.read_text(encoding="utf-8"))
        if scales["status"] != "passed" or set(scales["scales"]) != {"10000", "50000", "120000", "1000000"}:
            raise AssertionError("SIFT1M scale sweep is incomplete")
        checks.append("sift1m_scale_sweep")

    inat_encoding_path = ROOT / "results/v2.0/inat/encoding.json"
    if inat_encoding_path.exists():
        encoding = json.loads(inat_encoding_path.read_text(encoding="utf-8"))
        if encoding["status"] != "passed":
            raise AssertionError("iNaturalist encoding is incomplete")
        metadata_lock = json.loads(inat_lock_path.read_text(encoding="utf-8"))
        if set(encoding["splits"]) != set(metadata_lock["manifests"]):
            raise AssertionError("iNaturalist encoded split set differs from the frozen metadata lock")
        for split_name, split in encoding["splits"].items():
            for artifact in split["artifacts"].values():
                if sha256_file(resolve(artifact["path"])) != artifact["sha256"]:
                    raise AssertionError(f"iNaturalist embedding artifact hash mismatch: {artifact['path']}")
            expected_rows = jsonl(resolve(metadata_lock["manifests"][split_name]["path"]))
            expected = sorted((int(row["image_id"]), int(row["class_id"])) for row in expected_rows)
            image_ids = np.load(resolve(split["artifacts"]["image_ids"]["path"]))
            labels = np.load(resolve(split["artifacts"]["labels"]["path"]))
            vectors = np.load(resolve(split["artifacts"]["embeddings"]["path"]), mmap_mode="r")
            if vectors.shape != (len(expected), 384) or not np.all(np.isfinite(vectors)):
                raise AssertionError(f"iNaturalist {split_name} embedding shape/finiteness mismatch")
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-6):
                raise AssertionError(f"iNaturalist {split_name} embeddings are not unit normalized")
            actual = list(zip(map(int, image_ids), map(int, labels)))
            if actual != expected:
                raise AssertionError(f"iNaturalist {split_name} array/manifest alignment mismatch")
        checks.append("inat_embeddings")
    training_path = ROOT / "results/v2.0/inat/training.json"
    if training_path.exists():
        training = json.loads(training_path.read_text(encoding="utf-8"))
        if training["status"] != "passed" or len(training["long_tail"]) != 15:
            raise AssertionError("iNaturalist training result is incomplete")
        expected_methods = set(json.loads((ROOT / "configs/v2.0/experiment.json").read_text(encoding="utf-8"))["long_tail"]["methods"])
        if set(training["long_tail_summary"]) != expected_methods:
            raise AssertionError("iNaturalist long-tail summary method set mismatch")
        for method, summary in training["long_tail_summary"].items():
            if len(summary["seeds"]) != 5 or len(set(summary["seeds"])) != 5:
                raise AssertionError(f"iNaturalist {method} summary does not contain five unique seeds")
        model_path = resolve(training["balanced_probe"]["model_path"])
        if sha256_file(model_path) != training["balanced_probe"]["model_sha256"]:
            raise AssertionError("iNaturalist balanced probe checkpoint hash mismatch")
        checks.append("inat_training")

    open_set_path = ROOT / "results/v2.0/inat/open_set.json"
    if open_set_path.exists():
        open_set = json.loads(open_set_path.read_text(encoding="utf-8"))
        if open_set["status"] != "passed" or set(open_set["methods"]) != {
            "nearest_neighbor", "neighbour_agreement_margin", "temperature_max_probability", "negative_energy"
        }:
            raise AssertionError("iNaturalist open-set result is incomplete")
        if sha256_file(resolve(open_set["scores_path"])) != open_set["scores_sha256"]:
            raise AssertionError("iNaturalist open-set score artifact hash mismatch")
        if training_path.exists() and open_set["training_result_sha256"] != sha256_file(training_path):
            raise AssertionError("iNaturalist open-set result is not linked to authoritative training")
        checks.append("inat_open_set")

    monitoring_path = ROOT / "results/v2.0/inat/monitoring.json"
    if monitoring_path.exists():
        monitoring = json.loads(monitoring_path.read_text(encoding="utf-8"))
        if monitoring["status"] != "passed" or set(monitoring["evaluations"]) != {"test", "unknown_test"}:
            raise AssertionError("iNaturalist monitoring result is incomplete")
        if monitoring["encoding_sha256"] != sha256_file(inat_encoding_path):
            raise AssertionError("iNaturalist monitoring result is not linked to authoritative encoding")
        checks.append("inat_monitoring")

    result = {"status": "passed", "checks": checks}
    folder = ROOT / "results/v2.0/checks"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
