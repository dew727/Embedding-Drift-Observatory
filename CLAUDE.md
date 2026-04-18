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
    → Preprocessing (app/utils.py: preprocess())
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

| Module | Owner | Notes |
|---|---|---|
| `embeddings/` | Ryan | PCA from scratch, PyTorch Autoencoder, EmbeddingPipeline |
| `drift/` | Ethan Xu | All 4 drift generators + batch_simulator |
| `analysis/` | Quentin Liang | Geometric metrics on embedding space |
| `evaluation/` | Terrence Lin | Classifiers, ROC-AUC/F1/calibration, PerformanceTracker |
| `app/` | Stella | Streamlit app, all visualization components |

## Key Design Decisions

- **PCA is implemented from scratch** (`embeddings/pca.py`) — mean-centering, covariance matrix, SVD. Do not replace with `sklearn.decomposition.PCA`.
- **Embedder is frozen after baseline training** — `EmbeddingPipeline.fit()` is called once on baseline data; all drift batches go through `.transform()` only. This is intentional: refitting would mask the drift signal.
- **Drift is composable per batch** — `DriftConfig` in `batch_simulator.py` lets each batch have independent drift settings. Set a field to `0.0` / `None` to disable that drift type for a batch.
- **Modality-agnostic design** — the pipeline accepts any tabular numeric CSV; non-numeric columns are dropped in `app/utils.py:preprocess()`.
- **Batch-oriented temporal tracking** — `PerformanceTracker` and all geometry metrics are computed per batch, not per sample, to simulate production monitoring over time.

## Dataset

Primary: COVID-19 tabular dataset from Kaggle (`data/raw/covid19.csv`). The `data/` directory is gitignored — place datasets there locally.
