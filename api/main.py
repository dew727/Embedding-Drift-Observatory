"""FastAPI backend for the Embedding Drift Observatory.

Run with:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /upload          — upload a CSV, get back columns + label candidates
    POST /run             — run the full pipeline, get back serialized results
    GET  /health          — liveness check
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import uuid
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.dataset_processor import DatasetProcessor
from embeddings.pipeline import EmbeddingPipeline
from drift.batch_simulator import DriftConfig, simulate_batches
from analysis.cluster_tracker import ClusterTracker
from analysis.distance_metrics import mean_pairwise_shift
from analysis.neighbor_instability import neighbor_instability as compute_neighbor_instability
from analysis.drift_score import compute_drift_score
from evaluation.classifier import build_classifier, train_classifier
from evaluation.metrics import evaluate_batch
from evaluation.performance_tracker import PerformanceTracker


app = FastAPI(title="Embedding Drift Observatory API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for uploaded dataframes keyed by upload_id
_uploads: dict[str, pd.DataFrame] = {}
MAX_BASELINE_SCATTER_POINTS = 600
MAX_BATCH_SCATTER_POINTS = 240


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DriftConfigRequest(BaseModel):
    covariate_strength: float = 1.0
    prior_ratio: float | None = None
    concept_flip_rate: float = 0.0
    noise_std: float = 0.1
    missing_rate: float = 0.0


class RunRequest(BaseModel):
    upload_id: str
    label_col: str
    embed_method: str = "pca"       # "pca" | "autoencoder"
    n_components: int = 8
    clf_kind: str = "logistic"      # "logistic" | "mlp"
    n_batches: int = 6
    drift_config: DriftConfigRequest = DriftConfigRequest()
    n_clusters: int = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(arr) -> list:
    """Convert numpy array to nested Python list for JSON serialization."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def _emb_2d(emb: np.ndarray) -> list[list[float]]:
    """Return first 2 dimensions of embeddings for scatter plot."""
    return emb[:, :2].tolist()


def _sample_embedding_pairs(
    baseline_emb: np.ndarray,
    drift_emb: np.ndarray,
    max_points: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Sample matching baseline/drift points for response payloads."""
    if len(baseline_emb) != len(drift_emb):
        raise ValueError("baseline and drift embeddings must have the same length")

    if len(baseline_emb) <= max_points:
        indices = np.arange(len(baseline_emb))
    else:
        indices = np.random.choice(len(baseline_emb), size=max_points, replace=False)

    return _emb_2d(baseline_emb[indices]), _emb_2d(drift_emb[indices])


def _retrain_verdict(batch_results_data: list[dict]) -> dict:
    """Classify drift severity and produce a retraining recommendation."""
    scores = [b["drift_score"] for b in batch_results_data]
    accs   = [b["accuracy"] for b in batch_results_data if b["accuracy"] is not None and not (isinstance(b["accuracy"], float) and b["accuracy"] != b["accuracy"])]
    aucs   = [b["roc_auc"] for b in batch_results_data if b["roc_auc"] is not None and not (isinstance(b["roc_auc"], float) and b["roc_auc"] != b["roc_auc"])]
    ni_vals = [b["neighbor_instability"] for b in batch_results_data]

    peak_drift = max(scores)
    max_ni     = max(ni_vals)
    acc_drop   = (accs[0] - min(accs))  if len(accs) >= 2 else 0.0
    auc_drop   = (aucs[0] - min(aucs))  if len(aucs) >= 2 else 0.0

    signals = []
    if peak_drift > 0.4:
        signals.append(f"peak drift score {peak_drift:.2f}")
    if max_ni > 0.4:
        signals.append(f"neighbor instability {max_ni:.2f}")
    if acc_drop > 0.10:
        signals.append(f"accuracy dropped {acc_drop:.0%}")
    if auc_drop > 0.10:
        signals.append(f"AUC dropped {auc_drop:.2f}")

    geometry_alarm = peak_drift > 0.4 or max_ni > 0.4
    perf_alarm     = acc_drop > 0.10 or auc_drop > 0.10
    geometry_warn  = peak_drift > 0.2 or max_ni > 0.2
    perf_warn      = acc_drop > 0.05 or auc_drop > 0.05

    if geometry_alarm and perf_alarm:
        verdict = "retrain"
        label   = "Retraining Recommended"
        reason  = (
            "Both embedding geometry and classifier performance have degraded significantly. "
            f"Triggered by: {', '.join(signals)}. "
            "The model's internal representations no longer reflect the current data distribution — retraining on recent data is advised."
        )
    elif geometry_alarm and not perf_alarm:
        verdict = "monitor"
        label   = "Monitor Closely — Geometric Drift Detected"
        reason  = (
            "Embedding geometry has shifted substantially but classifier performance is still holding. "
            f"Triggered by: {', '.join(signals) if signals else 'elevated drift score'}. "
            "This is an early warning: performance degradation may follow. Schedule a retraining review."
        )
    elif geometry_warn or perf_warn:
        verdict = "monitor"
        label   = "Mild Drift — Continue Monitoring"
        reason  = (
            "Moderate drift detected in embedding space or performance metrics, but below retraining thresholds. "
            "Keep monitoring across future batches before taking action."
        )
    else:
        verdict = "stable"
        label   = "No Retraining Needed"
        reason  = (
            "Embedding geometry and classifier performance are stable across all batches. "
            "Distribution shift is within acceptable bounds."
        )

    return {
        "verdict": verdict,
        "label": label,
        "reason": reason,
        "signals": {
            "peak_drift_score":       round(peak_drift, 3),
            "max_neighbor_instability": round(max_ni, 3),
            "accuracy_drop":          round(acc_drop, 3),
            "auc_drop":               round(auc_drop, 3),
        },
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a CSV upload. Returns upload_id, all columns, and label candidates."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    upload_id = str(uuid.uuid4())
    _uploads[upload_id] = df

    proc = DatasetProcessor()
    label_candidates = proc.get_label_columns(df)

    return {
        "upload_id": upload_id,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "label_candidates": label_candidates,
    }


@app.post("/run")
def run(req: RunRequest) -> dict[str, Any]:
    """Run the full embedding drift pipeline and return serialized results."""
    if req.upload_id not in _uploads:
        raise HTTPException(status_code=404, detail="upload_id not found. Upload a CSV first.")

    df = _uploads[req.upload_id]

    # --- Preprocess ---
    proc = DatasetProcessor()
    try:
        X, y, feature_names = proc.preprocess(df, label_col=req.label_col)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    X_train, X_test, y_train, y_test = proc.split(X, y)

    # --- Embed (fit on baseline only) ---
    pipeline = EmbeddingPipeline(method=req.embed_method, n_components=req.n_components)
    pipeline.fit(X_train)
    baseline_emb = pipeline.transform(X_train)

    # --- Cluster tracker on baseline ---
    cluster_tracker = ClusterTracker(n_clusters=req.n_clusters)
    cluster_tracker.fit(baseline_emb.astype(np.float64))

    # --- Classifier ---
    clf = build_classifier(req.clf_kind)
    train_classifier(clf, baseline_emb, y_train)

    # --- Drift simulation ---
    drift_cfg = DriftConfig(
        covariate_strength=req.drift_config.covariate_strength,
        prior_ratio=req.drift_config.prior_ratio,
        concept_flip_rate=req.drift_config.concept_flip_rate,
        noise_std=req.drift_config.noise_std,
        missing_rate=req.drift_config.missing_rate,
    )
    drift_configs = [drift_cfg] * req.n_batches
    batch_size = max(50, len(X) // req.n_batches)
    batches = simulate_batches(X, y, req.n_batches, drift_configs, batch_size=batch_size)

    # --- Per-batch analysis ---
    perf_tracker = PerformanceTracker()
    batch_results = []

    for i, batch in enumerate(batches):
        drift_emb = pipeline.transform(batch.X).astype(np.float64)

        start    = (i * batch_size) % len(X)
        orig_idx = np.arange(start, start + batch_size) % len(X)
        orig_emb = pipeline.transform(X[orig_idx]).astype(np.float64)

        centroid_shifts = cluster_tracker.centroid_shift(drift_emb)
        reassign_rate   = cluster_tracker.reassignment_rate(orig_emb, drift_emb)
        nb_instability  = compute_neighbor_instability(orig_emb, drift_emb, k=5)
        mean_cos        = mean_pairwise_shift(orig_emb, drift_emb, metric="cosine")
        mean_euc        = mean_pairwise_shift(orig_emb, drift_emb, metric="euclidean")
        drift_score     = compute_drift_score(mean_cos, mean_euc, nb_instability,
                                              float(centroid_shifts.mean()), reassign_rate)

        try:
            metrics = evaluate_batch(clf, drift_emb, batch.y, batch_index=i)
            accuracy = metrics.accuracy
            f1       = metrics.f1
            roc_auc  = metrics.roc_auc
            frac_pos = _to_list(metrics.fraction_of_positives)
            mean_pred = _to_list(metrics.mean_predicted_value)
        except Exception:
            accuracy = f1 = roc_auc = None
            frac_pos = mean_pred = []

        if metrics:
            perf_tracker.log(metrics)

        sampled_baseline_2d, sampled_drift_2d = _sample_embedding_pairs(
            orig_emb,
            drift_emb,
            max_points=MAX_BATCH_SCATTER_POINTS,
        )

        batch_results.append({
            "index":               i,
            "emb_2d":              sampled_drift_2d,
            "baseline_emb_2d":     sampled_baseline_2d,
            "drift_score":         round(drift_score.score, 4),
            "mean_cosine_shift":   round(drift_score.mean_cosine_shift, 4),
            "mean_euclidean_shift": round(drift_score.mean_euclidean_shift, 4),
            "neighbor_instability": round(nb_instability, 4),
            "mean_centroid_shift": round(drift_score.mean_centroid_shift, 4),
            "reassignment_rate":   round(reassign_rate, 4),
            "centroid_shifts":     _to_list(centroid_shifts.round(4)),
            "accuracy":            round(accuracy, 4) if accuracy is not None else None,
            "f1":                  round(f1, 4) if f1 is not None else None,
            "roc_auc":             round(roc_auc, 4) if roc_auc is not None else None,
            "calibration": {
                "fraction_of_positives": frac_pos,
                "mean_predicted_value":  mean_pred,
            },
            "config": {
                "covariate_strength": batch.config.covariate_strength,
                "prior_ratio":        batch.config.prior_ratio,
                "concept_flip_rate":  batch.config.concept_flip_rate,
                "noise_std":          batch.config.noise_std,
                "missing_rate":       batch.config.missing_rate,
            },
        })

    summary = perf_tracker.summary()
    verdict = _retrain_verdict(batch_results)

    # Keep the baseline scatter compact so large uploads do not exhaust container memory.
    scatter_idx = np.random.choice(
        len(baseline_emb),
        size=min(MAX_BASELINE_SCATTER_POINTS, len(baseline_emb)),
        replace=False,
    )
    baseline_emb_2d_sample = _emb_2d(baseline_emb[scatter_idx])

    return {
        "verdict":         verdict,
        "summary":         summary,
        "feature_names":   feature_names,
        "n_train":         len(X_train),
        "n_test":          len(X_test),
        "baseline_emb_2d": baseline_emb_2d_sample,
        "batches":         batch_results,
    }
