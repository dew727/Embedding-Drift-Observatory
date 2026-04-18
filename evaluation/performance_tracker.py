"""Per-batch metric logger: accumulates BatchMetrics over time for trend visualization."""

import numpy as np
from evaluation.metrics import BatchMetrics


class PerformanceTracker:
    """Accumulates per-batch evaluation results for trend analysis and export."""

    def __init__(self):
        self._records: list[BatchMetrics] = []

    def log(self, metrics: BatchMetrics) -> None:
        self._records.append(metrics)

    def get_series(self, field: str) -> tuple[list[int], list[float]]:
        """Return (batch_indices, values) for a given metric field name."""
        indices = [r.batch_index for r in self._records]
        values = [getattr(r, field) for r in self._records]
        return indices, values

    def summary(self) -> dict:
        """Return mean and std for accuracy, f1, and roc_auc across all logged batches."""
        if not self._records:
            return {}
        results = {}
        for field in ("accuracy", "f1", "roc_auc"):
            values = np.array([getattr(r, field) for r in self._records])
            results[field] = {"mean": float(values.mean()), "std": float(values.std())}
        return results

    def reset(self) -> None:
        self._records.clear()
