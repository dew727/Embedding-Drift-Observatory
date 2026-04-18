"""Batch simulator: generates a sequence of drifted batches for temporal analysis."""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class DriftConfig:
    """Controls which drift types are applied and their intensities."""
    covariate_strength: float = 0.0     # shift_strength passed to covariate_shift
    prior_ratio: float | None = None    # target_ratio for prior_shift (None = disabled)
    concept_flip_rate: float = 0.0      # flip_rate for concept_drift
    noise_std: float = 0.0              # Gaussian noise std multiplier
    missing_rate: float = 0.0           # Missingness injection rate


@dataclass
class Batch:
    """A single temporal batch of (possibly drifted) data."""
    index: int
    X: np.ndarray
    y: np.ndarray
    config: DriftConfig = field(default_factory=DriftConfig)


def simulate_batches(
    X: np.ndarray,
    y: np.ndarray,
    n_batches: int,
    drift_configs: list[DriftConfig],
    batch_size: int = None,
    rng: np.random.Generator = None,
) -> list[Batch]:
    """Split data into n_batches and apply per-batch drift configs.

    Args:
        X: Full feature array.
        y: Full label array.
        n_batches: Number of temporal batches to produce.
        drift_configs: One DriftConfig per batch (length must equal n_batches).
        batch_size: Samples per batch. Defaults to len(X) // n_batches.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        List of Batch objects in temporal order.
    """
    from drift.covariate_shift import apply_covariate_shift
    from drift.prior_shift import apply_prior_shift
    from drift.concept_drift import apply_concept_drift
    from drift.noise_injection import inject_gaussian_noise, inject_missingness

    if len(drift_configs) != n_batches:
        raise ValueError("len(drift_configs) must equal n_batches")
    if rng is None:
        rng = np.random.default_rng()

    batch_size = batch_size or (len(X) // n_batches)
    batches = []

    for i, cfg in enumerate(drift_configs):
        start = (i * batch_size) % len(X)
        idx = np.arange(start, start + batch_size) % len(X)
        X_batch, y_batch = X[idx].copy(), y[idx].copy()

        if cfg.covariate_strength > 0:
            X_batch = apply_covariate_shift(X_batch, cfg.covariate_strength, rng=rng)
        if cfg.prior_ratio is not None:
            X_batch, y_batch = apply_prior_shift(X_batch, y_batch, cfg.prior_ratio, rng=rng)
        if cfg.concept_flip_rate > 0:
            X_batch, y_batch = apply_concept_drift(X_batch, y_batch, cfg.concept_flip_rate, rng=rng)
        if cfg.noise_std > 0:
            X_batch = inject_gaussian_noise(X_batch, cfg.noise_std, rng=rng)
        if cfg.missing_rate > 0:
            X_batch = inject_missingness(X_batch, cfg.missing_rate, rng=rng)

        batches.append(Batch(index=i, X=X_batch, y=y_batch, config=cfg))

    return batches
