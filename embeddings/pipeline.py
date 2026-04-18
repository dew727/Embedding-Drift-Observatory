"""Embedding pipeline: fit on baseline data, transform arbitrary batches.

The embedder is ALWAYS frozen after baseline training.
Never refit mid-pipeline — that would mask true drift signal.
"""

import numpy as np
import torch
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
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                return self._model.encode(X_tensor).numpy()

    def _train_autoencoder(self, X: np.ndarray) -> Autoencoder:
        input_dim = X.shape[1]
        latent_dim = self.n_components
        hidden_dims = self.kwargs.get("hidden_dims", [128, 64])
        lr = self.kwargs.get("lr", 1e-3)
        epochs = self.kwargs.get("epochs", 50)
        batch_size = self.kwargs.get("batch_size", 64)

        model = Autoencoder(input_dim, latent_dim, hidden_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        X_tensor = torch.tensor(X, dtype=torch.float32)

        model.train()
        for _ in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size]
                x_hat, _ = model(batch)
                loss = criterion(x_hat, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        return model
