"""Small, data-free Version 2.0 contracts suitable for CPU CI."""
from __future__ import annotations

import json
import threading
import tempfile
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import faiss
import numpy as np

from checks.v2_contracts import assert_no_self_match, assert_product_disjoint, sha256_file, verify_hash_manifest
from src.benchmarking.ann import overlap_recall, sharded_exact_l2
from src.retrieval.metrics import remove_self, retrieval_metrics
from src.retrieval.open_set import rejection_metrics, select_threshold
from src.service.api import Handler, status_payload
from src.training.long_tail import sample_ids
from src.training.metrics import classification_metrics, coverage_risk, open_set_metrics, reliability_bins
from src.monitoring.drift import centroid_cosine_shift, estimate_rbf_gamma, projection_psi, rbf_mmd
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads((ROOT / "configs/v2.0/experiment.json").read_text(encoding="utf-8"))
    assert config["long_tail"]["ratios"] == [10, 50]
    assert len(config["long_tail"]["seeds"]) >= 5 and len(set(config["long_tail"]["seeds"])) == len(config["long_tail"]["seeds"])
    assert config["exact"]["query_batch"] == 256 and config["exact"]["database_shard"] == 100000
    assert config["monitoring"]["bootstrap_partitions"] >= 10
    rng = np.random.default_rng(6207)
    database = rng.standard_normal((257, 16), dtype=np.float32)
    queries = rng.standard_normal((31, 16), dtype=np.float32)
    distances, ids = sharded_exact_l2(database, queries, 10, query_batch=7, database_shard=53)
    exact = faiss.IndexFlatL2(16)
    exact.add(database)
    expected_distances, expected_ids = exact.search(queries, 10)
    assert np.array_equal(ids, expected_ids)
    assert np.allclose(distances, expected_distances, atol=3e-5)
    assert overlap_recall(expected_ids, ids, 10) == 1.0

    hnsw = faiss.IndexHNSWFlat(16, 16)
    hnsw.hnsw.efConstruction = 100
    hnsw.hnsw.efSearch = 100
    hnsw.add(database)
    _, approximate = hnsw.search(queries, 10)
    assert overlap_recall(expected_ids, approximate, 10) >= 0.95

    query_ids = np.arange(6, dtype=np.int64)
    with_self = np.array([
        [0, 1, 2, 3], [1, 0, 2, 3], [2, 3, 0, 1],
        [3, 2, 4, 5], [4, 5, 2, 3], [5, 4, 2, 3],
    ])
    neighbours = remove_self(query_ids, with_self)
    assert_no_self_match(query_ids, neighbours)
    labels = np.array([0, 0, 1, 1, 2, 2])
    metrics = retrieval_metrics(labels, neighbours, ks=(1,), map_k=3)
    assert metrics["recall_at_1"] == 1.0 and metrics["map_at_3"] > 0.0
    assert_product_disjoint({1, 2}, {3, 4})

    candidate_ids = {class_id: list(range(class_id * 30, class_id * 30 + 30)) for class_id in range(10)}
    long_tail = sample_ids(candidate_ids, ratio=10, seed=6207)
    assert long_tail["realized_ratio"] == 10 and min(long_tail["counts"].values()) == 3
    try:
        sample_ids(candidate_ids, ratio=50, seed=6207)
        raise AssertionError("Infeasible exact 50:1 sampling must fail")
    except ValueError as error:
        assert "infeasible" in str(error)

    known_development = np.linspace(0.4, 1.0, 100)
    unknown_development = np.zeros(100)
    threshold = select_threshold(known_development, unknown_development, target_known_tpr=0.95)
    shifted_unknown = select_threshold(known_development, np.full(100, 0.41), target_known_tpr=0.95)
    assert shifted_unknown != threshold  # unknown-development data must affect selection
    brute_force = max(
        (
            (float(np.mean(known_development >= candidate)) + float(np.mean(unknown_development < candidate))) / 2,
            float(np.mean(unknown_development < candidate)),
            float(candidate),
        )
        for candidate in np.unique(np.concatenate((known_development, unknown_development)))
        if float(np.mean(known_development >= candidate)) + 1e-12 >= 0.95
    )[2]
    assert threshold == brute_force
    final = rejection_metrics(np.array([0.8, 0.9]), np.array([0.1, 0.2]), threshold)
    assert final["known_acceptance_rate"] == 1.0 and final["unknown_rejection_rate"] == 1.0
    probabilities = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    classification = classification_metrics(np.array([0, 1, 2, 0, 1, 2]), probabilities, {0: 10, 1: 5, 2: 1})
    assert classification["accuracy"] == 1.0 and 0 <= classification["ece"] <= 1
    reliability = reliability_bins(probabilities, np.array([0, 1, 2, 0, 1, 2]), bins=5)
    assert len(reliability) == 5 and sum(record["count"] for record in reliability) == len(probabilities)
    open_metrics = open_set_metrics(np.array([0.9, 0.8]), np.array([0.2, 0.1]), threshold=0.5)
    assert open_metrics["auroc_ood"] == 1.0 and open_metrics["unknown_rejection_rate"] == 1.0
    assert coverage_risk(np.array([0, 1]), np.array([0, 0]), np.array([0.9, 0.2]))[-1]["risk"] == 0.5
    reference = rng.standard_normal((40, 4), dtype=np.float32)
    projections = np.eye(4, dtype=np.float32)
    assert abs(centroid_cosine_shift(reference, reference)) < 1e-6
    assert projection_psi(reference, reference, projections) == 0.0
    gamma = estimate_rbf_gamma(reference, seed=6207)
    baseline_mmd = rbf_mmd(reference, reference, gamma=gamma)
    assert abs(baseline_mmd - rbf_mmd(reference.copy(), reference.copy(), gamma=gamma)) < 1e-12
    assert rbf_mmd(reference, reference + 3.0, gamma=gamma) > baseline_mmd

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        payload = root / "payload.txt"
        payload.write_text("locked", encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"files": {"payload.txt": sha256_file(payload)}}), encoding="utf-8")
        verify_hash_manifest(manifest, root)

    status = status_payload()
    assert status["status"] == "healthy"
    expected_status_artifacts = {
        "gpu_gate", "faiss_smoke", "sop_lock", "sop_retrieval", "sift1m_lock",
        "sift1m_benchmark", "sift1m_scale_sweep", "inat_image_lock", "inat_encoding",
        "inat_training", "inat_open_set", "inat_monitoring",
    }
    assert set(status["artifacts"]) == expected_status_artifacts
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/health", timeout=5) as response:
            assert response.status == 200 and json.load(response)["status"] == "healthy"
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/missing", timeout=5)
            raise AssertionError("Unknown API paths must return 404")
        except urllib.error.HTTPError as error:
            assert error.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    print(json.dumps({"status": "passed", "contracts": 14}, indent=2))


if __name__ == "__main__":
    main()
