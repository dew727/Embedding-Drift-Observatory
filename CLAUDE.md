# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Embedding Drift Observatory** monitors how ML model embeddings evolve under distribution shift. Instead of tracking surface-level metrics like accuracy alone, it analyzes the *geometry* of embedding spaces (cluster structure, pairwise distances, neighborhood stability) to detect deeper signals of model instability.

**Final Presentation:** April 27 – May 1, 2026

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app/main.py

# Run tests (once added)
pytest tests/

# Lint
ruff check . --fix
```

## Architecture

The system is a sequential ML pipeline with an interactive Streamlit frontend. Data flows top-to-bottom; each stage is a discrete module:

```
Data Ingestion (CSV upload via app/components/upload.py)
    → Preprocessing (data/dataset_processor.py OR app/utils.py: preprocess())
    → Data Splitting (app/utils.py: train_test_split_indices())
    → Embedding Learning (embeddings/pipeline.py: EmbeddingPipeline.fit() — fit on baseline only)
    → Baseline Modeling (evaluation/classifier.py)
    → Drift Simulation (drift/batch_simulator.py: simulate_batches())
    → Re-Embedding (embeddings/pipeline.py: EmbeddingPipeline.transform() on each batch)
    → Embedding Analysis (analysis/: distance_metrics, cluster_tracker, neighbor_instability)
    → Performance Evaluation (evaluation/metrics.py + performance_tracker.py — per batch)
    → Drift Quantification (analysis/drift_score.py: compute_drift_score())
    → Visualization (app/main.py + app/components/)
```

## Module Ownership

| Module | Owner | Status |
|---|---|---|
| `embeddings/pca.py` | Ryan | Done |
| `embeddings/autoencoder.py` | Ryan | Done |
| `embeddings/pipeline.py` | Ryan | Done |
| `data/dataset_processor.py` | Ethan Xu | Done |
| `drift/` (all 5 files) | Ethan Xu | Done |
| `analysis/` (all 4 files) | Quentin Liang | Done |
| `evaluation/` (all 3 files) | Terrence Lin | Done |
| `app/utils.py` | Stella | Done |
| `app/components/upload.py` | Stella | Done |
| `app/main.py` | Stella | Done |
| `app/components/embedding_viz.py` | Stella | Done |
| `app/components/cluster_viz.py` | Stella | Done |
| `app/components/retrieval_viz.py` | Stella | Done |
| `app/components/performance_viz.py` | Stella | Done |

## Key Design Decisions

- **PCA is implemented from scratch** (`embeddings/pca.py`) — mean-centering, covariance matrix, SVD. Do not replace with `sklearn.decomposition.PCA`.
- **Embedder is frozen after baseline training** — `EmbeddingPipeline.fit()` is called once on baseline data; all drift batches go through `.transform()` only. This is intentional: refitting would mask the drift signal.
- **Drift is composable per batch** — `DriftConfig` in `batch_simulator.py` lets each batch have independent drift settings. Set a field to `0.0` / `None` to disable that drift type for a batch.
- **Modality-agnostic design** — the pipeline accepts any tabular numeric CSV; non-numeric columns are dropped in `app/utils.py:preprocess()`.
- **Batch-oriented temporal tracking** — `PerformanceTracker` and all geometry metrics are computed per batch, not per sample, to simulate production monitoring over time.

## Pipeline Data Flow (implementation detail)

`app/main.py` runs the full pipeline on button click and caches results in `st.session_state["results"]`. The viz components receive pre-computed `list[BatchResult]` — they do no ML.

**`BatchResult`** (defined in `app/utils.py`) holds all per-batch outputs:
- `emb` — drifted batch embeddings
- `baseline_emb` — same samples embedded without drift (for comparison metrics)
- `metrics` — `BatchMetrics` (accuracy, F1, ROC-AUC, calibration)
- `centroid_shifts`, `reassignment_rate`, `neighbor_instability`
- `drift_score` — `DriftScoreResult` (weighted composite [0–1])
- `config` — the `DriftConfig` applied to this batch

**Known constraint:** `ClusterTracker` and `NearestNeighbors` require float64 inputs — `EmbeddingPipeline.transform()` returns float32. Cast with `.astype(np.float64)` is applied inside `ClusterTracker.fit/predict`. If passing embeddings directly to other sklearn estimators, cast first.

## Remaining tasks

- [ ] Place `data/raw/covid19.csv` locally and smoke-test with real data
- [ ] Add `data/raw/covid19.csv` path note to `.env.example`
- [ ] Add `pytest` tests for the drift generators and DatasetProcessor

## Dataset

Primary: COVID-19 tabular dataset from Kaggle (`data/raw/covid19.csv`). The `data/` directory is gitignored — place datasets there locally.
