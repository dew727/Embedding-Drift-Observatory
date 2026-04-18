"""Aggregate instability/drift score combining geometric embedding metrics."""

import numpy as np
from dataclasses import dataclass


@dataclass
class DriftScoreResult:
    mean_cosine_shift: float
    mean_euclidean_shift: float
    neighbor_instability: float
    mean_centroid_shift: float
    reassignment_rate: float
    score: float  # Weighted composite in [0, 1]


def compute_drift_score(
    mean_cosine_shift: float,
    mean_euclidean_shift: float,
    neighbor_instability: float,
    mean_centroid_shift: float,
    reassignment_rate: float,
    weights: dict[str, float] = None,
) -> DriftScoreResult:
    """Combine per-metric values into a single drift score.

    Each component is normalized before weighting; adjust `weights` to tune sensitivity.
    Default weights treat neighbor instability and reassignment rate most heavily.
    """
    if weights is None:
        weights = {
            "cosine": 0.15,
            "euclidean": 0.15,
            "neighbor": 0.30,
            "centroid": 0.15,
            "reassignment": 0.25,
        }

    # Simple min-max normalization using hard caps; components already in [0,1] except distances
    cosine_norm = min(mean_cosine_shift / 1.0, 1.0)       # cosine distance in [0, 2], cap at 1
    euclidean_norm = min(mean_euclidean_shift / 10.0, 1.0) # scale by expected magnitude
    centroid_norm = min(mean_centroid_shift / 10.0, 1.0)

    score = (
        weights["cosine"] * cosine_norm
        + weights["euclidean"] * euclidean_norm
        + weights["neighbor"] * neighbor_instability
        + weights["centroid"] * centroid_norm
        + weights["reassignment"] * reassignment_rate
    )

    return DriftScoreResult(
        mean_cosine_shift=mean_cosine_shift,
        mean_euclidean_shift=mean_euclidean_shift,
        neighbor_instability=neighbor_instability,
        mean_centroid_shift=mean_centroid_shift,
        reassignment_rate=reassignment_rate,
        score=float(np.clip(score, 0.0, 1.0)),
    )
