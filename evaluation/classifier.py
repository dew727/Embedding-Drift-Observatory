import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def build_classifier(kind: str = "logistic", **kwargs) -> ClassifierMixin:
    kind = kind.lower().strip()

    if kind == "logistic":
        default_kwargs = {
            "max_iter": 1000,
            "random_state": 42,
        }
        default_kwargs.update(kwargs)
        return LogisticRegression(**default_kwargs)

    elif kind == "mlp":
        default_kwargs = {
            "hidden_layer_sizes": (64, 32),
            "max_iter": 300,
            "random_state": 42,
            "early_stopping": True,
        }
        default_kwargs.update(kwargs)
        return MLPClassifier(**default_kwargs)

    else:
        raise ValueError(f"Unknown classifier kind '{kind}'. Use 'logistic' or 'mlp'.")


def train_classifier(
    clf: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()

    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}.")

    if y_train.ndim != 1:
        raise ValueError(f"y_train must be 1D after flattening, got shape {y_train.shape}.")

    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have the same number of samples. "
            f"Got {len(X_train)} and {len(y_train)}."
        )

    return clf.fit(X_train, y_train)


def predict_classifier(
    clf: ClassifierMixin,
    X: np.ndarray,
):
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")

    y_pred = clf.predict(X)

    y_prob = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_prob = proba[:, 1]
        else:
            y_prob = proba

    return y_pred, y_prob