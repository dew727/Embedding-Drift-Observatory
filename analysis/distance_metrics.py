"""Cosine and Euclidean distance utilities for comparing embedding batches."""

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance between rows of a and b (same shape)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return 1.0 - (a_norm * b_norm).sum(axis=1)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance between rows of a and b (same shape)."""
    return np.linalg.norm(a - b, axis=1)


def mean_pairwise_shift(baseline_emb: np.ndarray, drifted_emb: np.ndarray, metric: str = "cosine") -> float:
    """Average per-sample distance between baseline and drifted embeddings.

    Expects both arrays to have the same number of rows (matched samples).
    """
    if metric == "cosine":
        dists = cosine_distance(baseline_emb, drifted_emb)
    elif metric == "euclidean":
        dists = euclidean_distance(baseline_emb, drifted_emb)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'cosine' or 'euclidean'.")
    return float(dists.mean())
