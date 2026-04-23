import numpy as np
from evaluation.metrics import BatchMetrics


class PerformanceTracker:
    def __init__(self):
        self._records: list[BatchMetrics] = []

    def log(self, metrics: BatchMetrics) -> None:
        self._records.append(metrics)

    def get_series(self, field: str) -> tuple[list[int], list[float]]:
        indices = [r.batch_index for r in self._records]
        values = [getattr(r, field) for r in self._records]
        return indices, values

    def summary(self) -> dict:
        if not self._records:
            return {}

        results = {}
        for field in ("accuracy", "f1", "roc_auc"):
            values = np.array([getattr(r, field) for r in self._records], dtype=float)
            results[field] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }
        return results

    def reset(self) -> None:
        self._records.clear()