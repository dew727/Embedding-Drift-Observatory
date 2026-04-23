"""Embedding pipeline: fit on baseline data, transform arbitrary batches.

The embedder is ALWAYS frozen after baseline training.
Never refit mid-pipeline — that would mask true drift signal.
"""

import numpy as np
from embeddings.pca import PCA
from embeddings.autoencoder import Autoencoder


class EmbeddingPipeline:
    """Wraps PCA or Autoencoder behind a unified fit/transform interface."""

    METHODS = ("pca", "autoencoder")

    def __init__(self, method: str = "pca", n_components: int = 32, **kwargs):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        self.n_components = n_components
        self.kwargs = kwargs
        self._model = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "EmbeddingPipeline":
        """Fit embedder on baseline X. Call once — never call again after drift begins."""
        if self.method == "pca":
            self._model = PCA(n_components=self.n_components)
            self._model.fit(X)
        elif self.method == "autoencoder":
            self._model = self._train_autoencoder(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform a batch (baseline or drifted) into embedding space."""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fit before calling transform.")
        if self.method == "pca":
            return self._model.transform(X)
        elif self.method == "autoencoder":
            # X has shape (N, d). The custom NumPy autoencoder encode() method 
            # takes input of shape (d, N) and returns (h, cache).
            # We want the latent representation h, which has shape (k, N), 
            # and we must transpose it back to (N, k).
            h, _ = self._model.encode(X.T)
            return h.T

    def _train_autoencoder(self, X: np.ndarray) -> Autoencoder:
        model = Autoencoder(
            input_dim=X.shape[1],
            latent_dim=self.n_components,
            hidden_dims=self.kwargs.get("hidden_dims", [128, 64]),
            lr=self.kwargs.get("lr", 0.001)
        )
        
        # X has shape (N, d). The train method expects (d, N).
        model.train(
            X.T,
            epochs=self.kwargs.get("epochs", 50),
            batch_size=self.kwargs.get("batch_size", 64)
        )
        return model
