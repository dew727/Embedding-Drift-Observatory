"""Per-batch evaluation metrics: ROC-AUC, F1, accuracy, calibration."""

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
    # Calibration curve points
    fraction_of_positives: np.ndarray = None
    mean_predicted_value: np.ndarray = None


def evaluate_batch(
    clf: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    batch_index: int,
    n_calibration_bins: int = 10,
) -> BatchMetrics:
    """Evaluate a frozen classifier on one drifted batch.

    Args:
        clf: Trained (frozen) classifier.
        X: Batch embeddings.
        y: Batch labels.
        batch_index: Temporal index of this batch.
        n_calibration_bins: Bins for calibration curve.

    Returns:
        BatchMetrics for this batch.
    """
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    # ROC-AUC: handle binary vs. multiclass
    if y_proba.shape[1] == 2:
        auc = roc_auc_score(y, y_proba[:, 1])
    else:
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")

    # FIXED (Ryan): calibration_curve crashes when labels aren't {0,1} (e.g. {1,2}) — pass pos_label explicitly
    classes = clf.classes_
    pos_label = classes[1] if y_proba.shape[1] == 2 else None
    frac_pos, mean_pred = calibration_curve(
        y, y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.max(axis=1),
        n_bins=n_calibration_bins,
        pos_label=pos_label,
    )

    return BatchMetrics(
        batch_index=batch_index,
        accuracy=float(accuracy_score(y, y_pred)),
        f1=float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        roc_auc=float(auc),
        fraction_of_positives=frac_pos,
        mean_predicted_value=mean_pred,
    )
