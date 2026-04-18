"""Prior shift generator: resamples to change class proportions."""

import numpy as np


def apply_prior_shift(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 0.9,
    majority_class: int = 1,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample dataset so that `majority_class` makes up `target_ratio` of samples.

    Args:
        X: Features (n_samples, n_features).
        y: Labels (n_samples,).
        target_ratio: Desired fraction of the majority class (0 < target_ratio < 1).
        majority_class: The class label to oversample.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        X_shifted, y_shifted: Resampled arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    maj_idx = np.where(y == majority_class)[0]
    min_idx = np.where(y != majority_class)[0]

    n_total = len(y)
    n_maj = int(n_total * target_ratio)
    n_min = n_total - n_maj

    maj_sample = rng.choice(maj_idx, size=n_maj, replace=len(maj_idx) < n_maj)
    min_sample = rng.choice(min_idx, size=n_min, replace=len(min_idx) < n_min)

    idx = np.concatenate([maj_sample, min_sample])
    rng.shuffle(idx)
    return X[idx], y[idx]
