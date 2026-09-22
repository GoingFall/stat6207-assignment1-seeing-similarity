"""Deterministic embedding drift statistics."""
from __future__ import annotations

import numpy as np


def estimate_rbf_gamma(reference: np.ndarray, maximum_samples: int = 1000, seed: int = 0) -> float:
    """Freeze an RBF bandwidth from the reference distribution only."""
    reference = np.asarray(reference, dtype=np.float32)
    if len(reference) < 2:
        raise ValueError("RBF bandwidth estimation requires at least two reference samples")
    rng = np.random.default_rng(seed)
    sample = reference[rng.choice(len(reference), min(maximum_samples, len(reference)), replace=False)]
    distances = np.sum(sample * sample, axis=1)[:, None] + np.sum(sample * sample, axis=1)[None, :] - 2 * sample @ sample.T
    positive = distances[distances > 0]
    return 1.0 / float(np.median(positive)) if len(positive) else 1.0


def centroid_cosine_shift(reference: np.ndarray, current: np.ndarray) -> float:
    left, right = reference.mean(0), current.mean(0)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(1.0 - np.dot(left, right) / denominator) if denominator else 0.0


def projection_psi(reference: np.ndarray, current: np.ndarray, projections: np.ndarray, bins: int = 10) -> float:
    values = []
    for direction in projections:
        expected = reference @ direction
        observed = current @ direction
        edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        edges = np.unique(edges)
        expected_counts = np.histogram(expected, bins=edges)[0] / len(expected)
        observed_counts = np.histogram(observed, bins=edges)[0] / len(observed)
        expected_counts = np.clip(expected_counts, 1e-6, None)
        observed_counts = np.clip(observed_counts, 1e-6, None)
        values.append(float(np.sum((observed_counts - expected_counts) * np.log(observed_counts / expected_counts))))
    return float(np.mean(values))


def rbf_mmd(reference: np.ndarray, current: np.ndarray, gamma: float | None = None) -> float:
    if len(reference) < 2 or len(current) < 2:
        raise ValueError("Unbiased RBF-MMD requires at least two samples per set")
    if gamma is None:
        gamma = estimate_rbf_gamma(reference)
    def kernel(left, right):
        distances = np.sum(left * left, axis=1)[:, None] + np.sum(right * right, axis=1)[None, :] - 2 * left @ right.T
        return np.exp(-gamma * np.maximum(distances, 0))
    xx, yy, xy = kernel(reference, reference), kernel(current, current), kernel(reference, current)
    np.fill_diagonal(xx, 0); np.fill_diagonal(yy, 0)
    return float(xx.sum() / (len(reference) * (len(reference) - 1)) + yy.sum() / (len(current) * (len(current) - 1)) - 2 * xy.mean())


def drift_metrics(
    reference: np.ndarray,
    current: np.ndarray,
    projections: np.ndarray,
    bins: int,
    mmd_samples: int,
    seed: int,
    rbf_gamma: float | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    left = reference[rng.choice(len(reference), min(mmd_samples, len(reference)), replace=False)]
    right = current[rng.choice(len(current), min(mmd_samples, len(current)), replace=False)]
    return {
        "images": len(current),
        "norm_mean": float(np.linalg.norm(current, axis=1).mean()),
        "norm_std": float(np.linalg.norm(current, axis=1).std()),
        "centroid_cosine_shift": centroid_cosine_shift(reference, current),
        "projection_psi": projection_psi(reference, current, projections, bins),
        "rbf_mmd": rbf_mmd(left, right, gamma=rbf_gamma),
    }
