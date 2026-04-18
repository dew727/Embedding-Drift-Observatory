"""Concept drift generator: corrupts the feature-label relationship."""

import numpy as np


def apply_concept_drift(
    X: np.ndarray,
    y: np.ndarray,
    flip_rate: float = 0.2,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly flip a fraction of labels to simulate concept drift.

    Args:
        X: Features (n_samples, n_features) — returned unchanged.
        y: Labels (n_samples,).
        flip_rate: Fraction of labels to randomly reassign.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        X (unchanged), y_drifted: Labels with `flip_rate` fraction flipped.
    """
    if rng is None:
        rng = np.random.default_rng()
    y_drifted = y.copy()
    classes = np.unique(y)
    n_flip = int(len(y) * flip_rate)
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    for i in flip_idx:
        other = classes[classes != y[i]]
        y_drifted[i] = rng.choice(other)
    return X, y_drifted
