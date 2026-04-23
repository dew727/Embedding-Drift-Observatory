import numpy as np
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.base import ClassifierMixin


@dataclass
class BatchMetrics:
    batch_index: int
    accuracy: float
    f1: float
    roc_auc: float
    fraction_of_positives: np.ndarray = None
    mean_predicted_value: np.ndarray = None


def evaluate_batch(
    clf: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    batch_index: int,
    n_calibration_bins: int = 10,
) -> BatchMetrics:
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}.")
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of samples. Got {len(X)} and {len(y)}.")

    y_pred = clf.predict(X)

    if not hasattr(clf, "predict_proba"):
        raise ValueError("Classifier must support predict_proba for evaluation.")

    y_proba = clf.predict_proba(X)

    try:
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y, y_proba[:, 1])
        else:
            auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    frac_pos = None
    mean_pred = None

    if y_proba.shape[1] == 2:
        try:
            classes = clf.classes_
            pos_label = classes[1]
            frac_pos, mean_pred = calibration_curve(
                y,
                y_proba[:, 1],
                n_bins=n_calibration_bins,
                pos_label=pos_label,
            )
        except ValueError:
            frac_pos, mean_pred = None, None

    return BatchMetrics(
        batch_index=batch_index,
        accuracy=float(accuracy_score(y, y_pred)),
        f1=float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        roc_auc=float(auc),
        fraction_of_positives=frac_pos,
        mean_predicted_value=mean_pred,
    )