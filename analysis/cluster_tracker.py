"""k-means cluster tracking: centroid shift and cluster reassignment rate."""

import numpy as np
from sklearn.cluster import KMeans


class ClusterTracker:
    """Fits k-means on baseline embeddings, then tracks cluster stability over batches."""

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kmeans: KMeans = None
        self.baseline_labels_: np.ndarray = None
        self.baseline_centroids_: np.ndarray = None

    def fit(self, baseline_emb: np.ndarray) -> "ClusterTracker":
        """Fit k-means on baseline embeddings. Frozen after this call."""
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
        self.baseline_labels_ = self._kmeans.fit_predict(baseline_emb)
        self.baseline_centroids_ = self._kmeans.cluster_centers_.copy()
        return self

    def predict(self, emb: np.ndarray) -> np.ndarray:
        """Assign embeddings to baseline clusters (no refit)."""
        return self._kmeans.predict(emb)

    def centroid_shift(self, emb: np.ndarray) -> np.ndarray:
        """Mean distance each cluster centroid has moved (using current batch embeddings).

        Returns per-cluster centroid shift distances (n_clusters,).
        """
        new_centroids = np.array([
            emb[self.predict(emb) == k].mean(axis=0)
            if (self.predict(emb) == k).any() else self.baseline_centroids_[k]
            for k in range(self.n_clusters)
        ])
        return np.linalg.norm(new_centroids - self.baseline_centroids_, axis=1)

    def reassignment_rate(self, baseline_emb: np.ndarray, drifted_emb: np.ndarray) -> float:
        """Fraction of samples assigned to a different cluster after drift."""
        baseline_labels = self.predict(baseline_emb)
        drifted_labels = self.predict(drifted_emb)
        return float((baseline_labels != drifted_labels).mean())
