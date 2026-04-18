"""Noise injection: adds Gaussian noise or random missingness to features."""

import numpy as np


def inject_gaussian_noise(
    X: np.ndarray,
    noise_std: float = 0.1,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise scaled by `noise_std` * feature std."""
    if rng is None:
        rng = np.random.default_rng()
    scale = X.std(axis=0) * noise_std
    return X + rng.normal(0, scale, size=X.shape)


def inject_missingness(
    X: np.ndarray,
    missing_rate: float = 0.1,
    fill_value: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Randomly zero-out (or fill) entries to simulate missing data.

    Args:
        X: Input features.
        missing_rate: Fraction of all values to mark as missing.
        fill_value: Replacement value for missing entries (default 0.0).
        rng: Optional numpy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    X_noisy = X.copy()
    mask = rng.random(size=X.shape) < missing_rate
    X_noisy[mask] = fill_value
    return X_noisy
