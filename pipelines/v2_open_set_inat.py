"""Calibrate iNaturalist open-set rejection on development sets and evaluate once on held-out sets."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch
from scipy.optimize import minimize_scalar

from pipelines.prepare_data import ROOT
from src.retrieval.open_set import select_threshold
from src.training.metrics import coverage_risk, open_set_metrics
from src.training.metrics import expected_calibration_error, reliability_bins
from src.paths import resolve


EMBEDDINGS = ROOT / "data/v2.0/embeddings/inat_birds"
OUTPUT = ROOT / "results/v2.0/inat"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(EMBEDDINGS / f"{name}_embeddings.npy"), np.load(EMBEDDINGS / f"{name}_labels.npy")


def logits(model: torch.nn.Module, vectors: np.ndarray, batch_size: int) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(vectors), batch_size):
            chunks.append(model(torch.from_numpy(vectors[begin : begin + batch_size]).to("cuda")).cpu().numpy())
    return np.concatenate(chunks)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def temperature_nll(temperature: float, values: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.clip(softmax(values / temperature), 1e-12, 1.0)
    return float(-np.mean(np.log(probabilities[np.arange(len(labels)), labels])))


def knn_scores(index, train_labels: np.ndarray, vectors: np.ndarray, k: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    distances, neighbours = index.search(vectors, k)
    labels = train_labels[neighbours]
    predictions, agreement_margin = [], []
    for row in labels:
        counts = Counter(map(int, row))
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        predictions.append(ordered[0][0])
        second = ordered[1][1] if len(ordered) > 1 else 0
        agreement_margin.append((ordered[0][1] - second) / k)
    return {
        "nearest_neighbor": distances[:, 0].astype(np.float64),
        "neighbour_agreement_margin": np.asarray(agreement_margin, dtype=np.float64),
    }, np.asarray(predictions, dtype=np.int64)


def main(revision: bool = False) -> None:
    global EMBEDDINGS, OUTPUT
    if revision:
        EMBEDDINGS = ROOT / 'data/v2.1/embeddings/inat_birds'
        OUTPUT = ROOT / 'results/v2.1/inat'
        if (OUTPUT/'open_set.json').exists(): raise RuntimeError('Revision open-set result exists')
    training_path = OUTPUT / "training.json"
    training_result = json.loads(training_path.read_text(encoding="utf-8"))
    if training_result["status"] != "passed":
        raise RuntimeError("Passed balanced-probe training is required")
    config_path = ROOT / ('configs/v2.1/experiment.json' if revision else 'configs/v2.0/experiment.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))['inat_training']
    batch_size = config["linear_probe"]["batch_size"]
    train_x, train_labels = load("train")
    validation_x, validation_labels = load("validation")
    test_x, test_labels = load("test")
    unknown_development_x, _ = load("unknown_development")
    unknown_test_x, _ = load("unknown_test")

    checkpoint = torch.load(resolve(training_result["balanced_probe"]["model_path"]), map_location="cpu", weights_only=False)
    class_ids = np.asarray(checkpoint["class_ids"], dtype=np.int64)
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    validation_y = np.array([class_to_index[int(value)] for value in validation_labels], dtype=np.int64)
    test_y = np.array([class_to_index[int(value)] for value in test_labels], dtype=np.int64)
    train_y = np.array([class_to_index[int(value)] for value in train_labels], dtype=np.int64)
    model = torch.nn.Linear(train_x.shape[1], len(class_ids)).to("cuda")
    model.load_state_dict(checkpoint["state_dict"])

    validation_logits = logits(model, validation_x, batch_size)
    test_logits = logits(model, test_x, batch_size)
    unknown_development_logits = logits(model, unknown_development_x, batch_size)
    unknown_test_logits = logits(model, unknown_test_x, batch_size)
    lower, upper = config["calibration"]["temperature_bounds"]
    optimization = minimize_scalar(
        temperature_nll, bounds=(lower, upper), method="bounded", args=(validation_logits, validation_y),
        options={"xatol": 1e-5},
    )
    if not optimization.success:
        raise RuntimeError(f"Temperature optimization failed: {optimization.message}")
    temperature = float(optimization.x)
    validation_uncalibrated = softmax(validation_logits)
    validation_calibrated = softmax(validation_logits / temperature)
    test_uncalibrated = softmax(test_logits)
    test_calibrated = softmax(test_logits / temperature)
    calibration = {
        "fit_set": "known_validation_only",
        "temperature": temperature,
        "validation": {
            "nll_before": temperature_nll(1.0, validation_logits, validation_y),
            "nll_after": temperature_nll(temperature, validation_logits, validation_y),
            "ece_before": expected_calibration_error(validation_uncalibrated, validation_y, config["calibration"]["ece_bins"]),
            "ece_after": expected_calibration_error(validation_calibrated, validation_y, config["calibration"]["ece_bins"]),
        },
        "test": {
            "nll_before": temperature_nll(1.0, test_logits, test_y),
            "nll_after": temperature_nll(temperature, test_logits, test_y),
            "ece_before": expected_calibration_error(test_uncalibrated, test_y, config["calibration"]["ece_bins"]),
            "ece_after": expected_calibration_error(test_calibrated, test_y, config["calibration"]["ece_bins"]),
            "reliability_before": reliability_bins(test_uncalibrated, test_y, config["calibration"]["ece_bins"]),
            "reliability_after": reliability_bins(test_calibrated, test_y, config["calibration"]["ece_bins"]),
        },
    }

    index = faiss.IndexFlatIP(train_x.shape[1])
    index.add(train_x)
    validation_knn, _ = knn_scores(index, train_y, validation_x, config["frozen_knn_k"])
    test_knn, test_knn_predictions = knn_scores(index, train_y, test_x, config["frozen_knn_k"])
    unknown_development_knn, _ = knn_scores(index, train_y, unknown_development_x, config["frozen_knn_k"])
    unknown_test_knn, _ = knn_scores(index, train_y, unknown_test_x, config["frozen_knn_k"])

    validation_probabilities = validation_calibrated
    test_probabilities = test_calibrated
    unknown_development_probabilities = softmax(unknown_development_logits / temperature)
    unknown_test_probabilities = softmax(unknown_test_logits / temperature)
    score_sets = {
        "nearest_neighbor": (validation_knn["nearest_neighbor"], unknown_development_knn["nearest_neighbor"], test_knn["nearest_neighbor"], unknown_test_knn["nearest_neighbor"], test_knn_predictions),
        "neighbour_agreement_margin": (validation_knn["neighbour_agreement_margin"], unknown_development_knn["neighbour_agreement_margin"], test_knn["neighbour_agreement_margin"], unknown_test_knn["neighbour_agreement_margin"], test_knn_predictions),
        "temperature_max_probability": (validation_probabilities.max(1), unknown_development_probabilities.max(1), test_probabilities.max(1), unknown_test_probabilities.max(1), test_probabilities.argmax(1)),
        "negative_energy": (
            temperature * np.log(np.exp(validation_logits / temperature - (validation_logits / temperature).max(1, keepdims=True)).sum(1)) + validation_logits.max(1),
            temperature * np.log(np.exp(unknown_development_logits / temperature - (unknown_development_logits / temperature).max(1, keepdims=True)).sum(1)) + unknown_development_logits.max(1),
            temperature * np.log(np.exp(test_logits / temperature - (test_logits / temperature).max(1, keepdims=True)).sum(1)) + test_logits.max(1),
            temperature * np.log(np.exp(unknown_test_logits / temperature - (unknown_test_logits / temperature).max(1, keepdims=True)).sum(1)) + unknown_test_logits.max(1),
            test_probabilities.argmax(1),
        ),
    }
    if set(score_sets) != set(config["open_set"]["scores"]):
        raise AssertionError("Open-set score methods differ from preregistration")
    result = {
        "status": "running", "training_result_sha256": hash_file(training_path), "calibration": calibration,
        "unknown_scope": config["open_set"]["unknown_scope"], "far_ood_status": config["open_set"]["far_ood_status"], "methods": {},
    }
    score_artifacts = {}
    if revision:
        result['config_sha256'] = hash_file(config_path)
        result['encoding_sha256'] = hash_file(OUTPUT/'encoding.json')
        score_artifacts.update(validation_logits=validation_logits, validation_labels=validation_y, test_logits=test_logits, test_labels=test_y, test_knn_predictions=test_knn_predictions)
    for name, (known_development, unknown_development, known_test, unknown_test, predictions) in score_sets.items():
        threshold = select_threshold(known_development, unknown_development, config["open_set"]["target_known_tpr"])
        development_metrics = open_set_metrics(known_development, unknown_development, threshold)
        metrics = open_set_metrics(known_test, unknown_test, threshold)
        accepted = known_test >= threshold
        metrics["known_accuracy_conditional_on_acceptance"] = float(np.mean(predictions[accepted] == test_y[accepted])) if np.any(accepted) else None
        metrics["coverage_risk"] = coverage_risk(test_y, predictions, known_test)
        result["methods"][name] = {
            "threshold_selection_sets": ["known_validation", "unknown_development"],
            "final_evaluation_sets": ["known_test", "unknown_test"],
            "threshold": threshold,
            "development": development_metrics,
            "test": metrics,
        }
        score_artifacts[f"{name}_known_development"] = known_development
        score_artifacts[f"{name}_unknown_development"] = unknown_development
        score_artifacts[f"{name}_known_test"] = known_test
        score_artifacts[f"{name}_unknown_test"] = unknown_test
        if revision: score_artifacts[f'{name}_test_predictions'] = predictions
    scores_path = OUTPUT / "open_set_scores.npz"
    np.savez_compressed(scores_path, **score_artifacts)
    result["scores_path"] = scores_path.relative_to(ROOT).as_posix()
    result["scores_sha256"] = hash_file(scores_path)
    result["status"] = "passed"
    (OUTPUT / "open_set.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "calibration": calibration, "methods": {name: data["test"] for name, data in result["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
