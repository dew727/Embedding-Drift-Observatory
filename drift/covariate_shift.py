"""Covariate shift generator: perturbs feature distributions while preserving labels."""

import numpy as np


def apply_covariate_shift(
    X: np.ndarray,
    shift_strength: float = 1.0,
    feature_indices: list[int] = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Shift feature means by `shift_strength` standard deviations.

    Args:
        X: Input features (n_samples, n_features).
        shift_strength: Magnitude of mean shift in units of each feature's std.
        feature_indices: Which features to shift. Defaults to all.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        X_shifted: Shifted copy of X.
    """
    if rng is None:
        rng = np.random.default_rng()
    X_shifted = X.copy()
    cols = feature_indices if feature_indices is not None else list(range(X.shape[1]))
    stds = X[:, cols].std(axis=0)
    X_shifted[:, cols] += shift_strength * stds * rng.choice([-1, 1], size=len(cols))
    return X_shifted
