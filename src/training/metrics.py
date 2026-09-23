"""Long-tail classification, calibration and open-set metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve


def reliability_bins(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> list[dict]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.ndim != 2 or len(probabilities) != len(labels) or bins < 2:
        raise ValueError("Expected N×C probabilities, N labels and at least two bins")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    records = []
    for index in range(bins):
        include = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index < bins - 1 else confidence <= edges[index + 1]
        )
        records.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(np.sum(include)),
            "mean_confidence": float(np.mean(confidence[include])) if np.any(include) else None,
            "accuracy": float(np.mean(correct[include])) if np.any(include) else None,
        })
    return records


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.ndim != 2 or len(probabilities) != len(labels) or bins < 2:
        raise ValueError("Expected N×C probabilities, N labels and at least two bins")
    records = reliability_bins(probabilities, labels, bins)
    return float(sum(
        record["count"] / len(labels) * abs(record["accuracy"] - record["mean_confidence"])
        for record in records if record["count"]
    ))


def class_groups(train_counts: dict[int, int]) -> dict[str, set[int]]:
    """Assign deterministic head/medium/tail groups by count then class ID tertiles."""
    if len(train_counts) < 3:
        raise ValueError("Head/medium/tail metrics require at least three classes")
    ordered = sorted(train_counts, key=lambda class_id: (-train_counts[class_id], class_id))
    parts = np.array_split(np.asarray(ordered, dtype=np.int64), 3)
    return {name: set(map(int, values)) for name, values in zip(("head", "medium", "tail"), parts)}


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    train_counts: dict[int, int],
    ece_bins: int = 15,
) -> dict:
    labels = np.asarray(labels)
    predictions = np.asarray(probabilities).argmax(axis=1)
    groups = class_groups(train_counts)
    result = {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probabilities, labels, bins=ece_bins),
        "groups": {},
    }
    for name, classes in groups.items():
        include = np.isin(labels, list(classes))
        group_labels = sorted(classes)
        result["groups"][name] = {
            "classes": len(classes),
            "images": int(np.sum(include)),
            "macro_f1": float(f1_score(labels, predictions, labels=group_labels, average="macro", zero_division=0)),
            "recall": float(np.mean([
                np.mean(predictions[labels == class_id] == class_id)
                for class_id in group_labels if np.any(labels == class_id)
            ])),
        }
    return result


def open_set_metrics(known_scores: np.ndarray, unknown_scores: np.ndarray, threshold: float) -> dict:
    """Evaluate a known-high score; OOD is the positive class for AUROC/AUPR/FPR95."""
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    ood_labels = np.concatenate((np.zeros(len(known_scores), dtype=np.int8), np.ones(len(unknown_scores), dtype=np.int8)))
    ood_scores = -np.concatenate((known_scores, unknown_scores))
    fpr, tpr, _ = roc_curve(ood_labels, ood_scores)
    at_95 = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc_ood": float(roc_auc_score(ood_labels, ood_scores)),
        "aupr_ood": float(average_precision_score(ood_labels, ood_scores)),
        "fpr_at_95_tpr_ood": float(fpr[at_95[0]]) if len(at_95) else 1.0,
        "known_coverage": float(np.mean(known_scores >= threshold)),
        "unknown_rejection_rate": float(np.mean(unknown_scores < threshold)),
        "threshold": float(threshold),
    }


def coverage_risk(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray, points: int = 21) -> list[dict]:
    labels, predictions, scores = map(np.asarray, (labels, predictions, scores))
    order = np.argsort(-scores, kind="stable")
    correct = (labels[order] == predictions[order]).astype(np.float64)
    curve = []
    for coverage in np.linspace(0.05, 1.0, points):
        accepted = max(1, int(np.ceil(coverage * len(labels))))
        curve.append({
            "coverage": accepted / len(labels),
            "risk": 1.0 - float(np.mean(correct[:accepted])),
            "accepted": accepted,
        })
    return curve
