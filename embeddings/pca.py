"""PCA implemented from scratch using mean-centering and SVD/eigendecomposition."""

import numpy as np


class PCA:
    """Principal Component Analysis implemented from scratch.

    Uses mean-centering followed by SVD on the covariance matrix.
    Do NOT replace with sklearn.decomposition.PCA — from-scratch is a project requirement.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # shape: (n_components, n_features)
        self.explained_variance_ = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit PCA on baseline data X (n_samples, n_features)."""
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix and eigendecomposition via SVD
        _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        self.explained_variance_ = (s**2) / (len(X) - 1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X into the fitted PCA space."""
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
