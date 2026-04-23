# Frontend Integration Guide

This document is for the React frontend developer / agent building the UI for the Embedding Drift Observatory.

## Starting the backend

```bash
# From the project root: /Embedding-Drift-Observatory/
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Interactive API docs (auto-generated): http://localhost:8000/docs

CORS is open to all origins — no proxy config needed in React dev.

---

## Endpoints

### GET /health
Liveness check.

**Response:**
```json
{ "status": "ok" }
```

---

### POST /upload
Upload a CSV file. Returns metadata and candidate label columns so the user can pick one.

**Request:** multipart/form-data, field name `file`.

**Response:**
```json
{
  "upload_id": "uuid-string",
  "n_rows": 768,
  "n_cols": 9,
  "columns": ["Pregnancies", "Glucose", "BMI", "Outcome", ...],
  "label_candidates": ["Outcome", "BMI", ...]
}
```

- `upload_id` — pass this to `/run`
- `label_candidates` — columns with fewer than 50 unique values; show these in a dropdown for the user to pick the label

---

### POST /run
Run the full embedding drift pipeline. Takes ~5–30 seconds depending on dataset size and number of batches.

**Request body (JSON):**
```json
{
  "upload_id": "uuid-string",
  "label_col": "Outcome",
  "embed_method": "pca",
  "n_components": 8,
  "clf_kind": "logistic",
  "n_batches": 6,
  "n_clusters": 4,
  "drift_config": {
    "covariate_strength": 1.0,
    "noise_std": 0.1,
    "concept_flip_rate": 0.0,
    "prior_ratio": null,
    "missing_rate": 0.0
  }
}
```

**Parameter reference:**

| Field | Type | Options | Description |
|---|---|---|---|
| `upload_id` | string | — | From `/upload` |
| `label_col` | string | any column name | Target/label column |
| `embed_method` | string | `"pca"`, `"autoencoder"` | Embedding method |
| `n_components` | int | 2–64 | Embedding dimensions (latent size) |
| `clf_kind` | string | `"logistic"`, `"mlp"` | Baseline classifier |
| `n_batches` | int | 2–20 | Number of temporal drift batches |
| `n_clusters` | int | 2–16 | k-means clusters for geometry analysis |
| `drift_config.covariate_strength` | float | 0.0–3.0 | Feature mean shift magnitude |
| `drift_config.noise_std` | float | 0.0–1.0 | Gaussian noise scale |
| `drift_config.concept_flip_rate` | float | 0.0–0.5 | Fraction of labels randomly flipped |
| `drift_config.prior_ratio` | float or null | 0.5–0.99 | Oversample majority class to this ratio; null = disabled |
| `drift_config.missing_rate` | float | 0.0–0.5 | Fraction of feature values zeroed out |

---

**Response:**
```json
{
  "verdict": {
    "verdict": "retrain",
    "label": "Retraining Recommended",
    "reason": "Both embedding geometry and classifier performance have degraded...",
    "signals": {
      "peak_drift_score": 0.641,
      "max_neighbor_instability": 0.84,
      "accuracy_drop": 0.12,
      "auc_drop": 0.18
    }
  },
  "summary": {
    "accuracy": { "mean": 0.77, "std": 0.29 },
    "f1":       { "mean": 0.77, "std": 0.31 },
    "roc_auc":  { "mean": 0.95, "std": 0.08 }
  },
  "feature_names": ["Pregnancies", "Glucose", "BMI", ...],
  "n_train": 614,
  "n_test": 154,
  "baseline_emb_2d": [[-2.27, 0.63], [0.86, -2.16], ...],
  "batches": [
    {
      "index": 0,
      "drift_score": 0.004,
      "mean_cosine_shift": 0.001,
      "mean_euclidean_shift": 0.003,
      "neighbor_instability": 0.01,
      "mean_centroid_shift": 0.002,
      "reassignment_rate": 0.04,
      "centroid_shifts": [0.1, 0.2, 0.15, 0.08],
      "accuracy": 0.996,
      "f1": 0.996,
      "roc_auc": 1.0,
      "calibration": {
        "fraction_of_positives": [0.05, 0.2, 0.5, 0.8, 0.95],
        "mean_predicted_value":  [0.04, 0.19, 0.51, 0.79, 0.94]
      },
      "emb_2d": [[-1.1, 0.4], [0.3, -0.9], ...],
      "baseline_emb_2d": [[-1.0, 0.5], [0.4, -0.8], ...],
      "config": {
        "covariate_strength": 1.0,
        "noise_std": 0.1,
        "concept_flip_rate": 0.0,
        "prior_ratio": null,
        "missing_rate": 0.0
      }
    }
  ]
}
```

---

## Key data notes for rendering

**Verdict banner** — `verdict.verdict` is one of `"stable"`, `"monitor"`, `"retrain"`. Suggest green/yellow/red respectively. Display `verdict.label` as the heading and `verdict.reason` as the body text.

**Embedding scatter plot** — `baseline_emb_2d` gives up to 2000 baseline points (x, y). Each batch has `emb_2d` (drifted) and `baseline_emb_2d` (same samples, no drift) for comparison. Plot both overlaid — baseline in one color, drifted in another.

**Drift score over time** — `batches[i].drift_score` is a float in [0, 1]. Plot as a line chart across batch indices to show drift progression.

**Performance over time** — `batches[i].accuracy`, `batches[i].f1`, `batches[i].roc_auc` per batch. Line chart showing degradation.

**Centroid shifts** — `batches[i].centroid_shifts` is an array of per-cluster shift distances (length = n_clusters). Bar chart per batch.

**Calibration curve** — `batches[i].calibration.fraction_of_positives` vs `mean_predicted_value`. A perfectly calibrated model follows the diagonal.

**Neighbor instability & reassignment rate** — scalar per batch, plot as line charts alongside drift score.

---

## Recommended UI layout

```
[Sidebar]
  - CSV upload
  - Label column dropdown (populated after upload)
  - Embedding method (pca / autoencoder)
  - Dimensions slider (2–64)
  - Classifier (logistic / mlp)
  - Number of batches slider (2–20)
  - Drift settings (collapsible)
  - Run button

[Main panel]
  - Verdict banner (full width, color-coded)
  - Summary metrics row (mean accuracy / f1 / roc-auc)
  - Tabs:
      1. Embedding Space    — scatter: baseline vs each batch
      2. Cluster Dynamics   — centroid shifts + reassignment rate over batches
      3. Retrieval Stability — neighbor instability over batches
      4. Performance        — accuracy / f1 / auc line charts + calibration
```
