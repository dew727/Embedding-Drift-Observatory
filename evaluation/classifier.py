"""Baseline classifiers: Logistic Regression and a simple MLP."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.base import ClassifierMixin


def build_classifier(kind: str = "logistic", **kwargs) -> ClassifierMixin:
    """Factory for baseline classifiers.

    Args:
        kind: 'logistic' or 'mlp'.
        **kwargs: Passed to the underlying sklearn constructor.
    """
    if kind == "logistic":
        return LogisticRegression(max_iter=1000, **kwargs)
    elif kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, **kwargs)
    else:
        raise ValueError(f"Unknown classifier kind '{kind}'. Use 'logistic' or 'mlp'.")


def train_classifier(
    clf: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Fit classifier on baseline training embeddings."""
    return clf.fit(X_train, y_train)
