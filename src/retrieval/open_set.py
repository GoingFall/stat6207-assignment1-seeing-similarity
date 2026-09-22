"""Development-only threshold selection and open-set evaluation metrics."""
from __future__ import annotations

import numpy as np


def select_threshold(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    target_known_tpr: float = 0.95,
) -> float:
    """Select the highest-score-is-known threshold using development data only."""
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    if known_scores.ndim != 1 or unknown_scores.ndim != 1 or not len(known_scores) or not len(unknown_scores):
        raise ValueError("Known and unknown development scores must be non-empty vectors")
    if not 0 < target_known_tpr <= 1:
        raise ValueError("target_known_tpr must be in (0, 1]")
    if not np.all(np.isfinite(known_scores)) or not np.all(np.isfinite(unknown_scores)):
        raise ValueError("Scores must be finite")
    candidates = np.unique(np.concatenate((known_scores, unknown_scores)))
    known_sorted = np.sort(known_scores)
    unknown_sorted = np.sort(unknown_scores)
    known_tpr = (len(known_sorted) - np.searchsorted(known_sorted, candidates, side="left")) / len(known_sorted)
    unknown_tnr = np.searchsorted(unknown_sorted, candidates, side="left") / len(unknown_sorted)
    feasible = known_tpr + 1e-12 >= target_known_tpr
    if not np.any(feasible):
        raise RuntimeError("No threshold satisfies the known-development coverage target")
    feasible_candidates = candidates[feasible]
    feasible_unknown_tnr = unknown_tnr[feasible]
    feasible_balanced_accuracy = (known_tpr[feasible] + feasible_unknown_tnr) / 2.0
    # Prefer stronger unknown rejection, then the higher threshold, on exact ties.
    order = np.lexsort((feasible_candidates, feasible_unknown_tnr, feasible_balanced_accuracy))
    return float(feasible_candidates[order[-1]])


def rejection_metrics(known_scores: np.ndarray, unknown_scores: np.ndarray, threshold: float) -> dict:
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    return {
        "known_acceptance_rate": float(np.mean(known_scores >= threshold)),
        "unknown_rejection_rate": float(np.mean(unknown_scores < threshold)),
        "threshold": float(threshold),
    }
