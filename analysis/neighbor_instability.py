"""Nearest-neighbor instability: measures how much the neighborhood of each point changes."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def neighbor_instability(
    baseline_emb: np.ndarray,
    drifted_emb: np.ndarray,
    k: int = 5,
) -> float:
    """Fraction of k-nearest neighbors that change between baseline and drifted embeddings.

    Both arrays must have the same number of rows (matched samples).

    Args:
        baseline_emb: Baseline embeddings (n_samples, dim).
        drifted_emb: Drifted embeddings (n_samples, dim).
        k: Number of neighbors.

    Returns:
        Instability score in [0, 1]: 0 = no change, 1 = all neighbors changed.
    """
    nn_base = NearestNeighbors(n_neighbors=k + 1).fit(baseline_emb)
    nn_drift = NearestNeighbors(n_neighbors=k + 1).fit(drifted_emb)

    _, idx_base = nn_base.kneighbors(baseline_emb)
    _, idx_drift = nn_drift.kneighbors(drifted_emb)

    # Exclude self (index 0) and compare neighbor sets
    instabilities = []
    for i in range(len(baseline_emb)):
        base_set = set(idx_base[i, 1:])
        drift_set = set(idx_drift[i, 1:])
        instabilities.append(len(base_set - drift_set) / k)

    return float(np.mean(instabilities))
