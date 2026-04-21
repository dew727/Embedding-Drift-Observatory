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
| `embeddings/autoencoder.py` | Ryan | **Empty — needs implementation** |
| `embeddings/pipeline.py` | Ryan | Done (blocked by autoencoder) |
| `data/dataset_processor.py` | Ethan Xu | **Empty — needs implementation** |
| `drift/` (all 5 files) | Ethan Xu | Done |
| `analysis/` (all 4 files) | Quentin Liang | Done |
| `evaluation/` (all 3 files) | Terrence Lin | Done |
| `app/utils.py` | Stella | Done |
| `app/components/upload.py` | Stella | Done |
| `app/main.py` | Stella | **Scaffold only — pipeline not wired** |
| `app/components/embedding_viz.py` | Stella | **Placeholder — needs implementation** |
| `app/components/cluster_viz.py` | Stella | **Placeholder — needs implementation** |
| `app/components/retrieval_viz.py` | Stella | **Placeholder — needs implementation** |
| `app/components/performance_viz.py` | Stella | **Placeholder — needs implementation** |

## Key Design Decisions

- **PCA is implemented from scratch** (`embeddings/pca.py`) — mean-centering, covariance matrix, SVD. Do not replace with `sklearn.decomposition.PCA`.
- **Embedder is frozen after baseline training** — `EmbeddingPipeline.fit()` is called once on baseline data; all drift batches go through `.transform()` only. This is intentional: refitting would mask the drift signal.
- **Drift is composable per batch** — `DriftConfig` in `batch_simulator.py` lets each batch have independent drift settings. Set a field to `0.0` / `None` to disable that drift type for a batch.
- **Modality-agnostic design** — the pipeline accepts any tabular numeric CSV; non-numeric columns are dropped in `app/utils.py:preprocess()`.
- **Batch-oriented temporal tracking** — `PerformanceTracker` and all geometry metrics are computed per batch, not per sample, to simulate production monitoring over time.

## Implementation Plan (remaining work)

### Ethan — `data/dataset_processor.py` (unblocked, do first)

Implement a `DatasetProcessor` class that handles the full data preparation pipeline. It should be usable independently of Streamlit (no `st.*` calls):

```python
class DatasetProcessor:
    def load(self, path_or_df) -> pd.DataFrame         # accept filepath or DataFrame
    def preprocess(self, df, label_col) -> (X, y, feature_names)
        # Drop non-numeric, fill NaNs with column means, StandardScaler normalization
    def split(self, X, y, test_ratio=0.2, seed=42) -> (X_train, X_test, y_train, y_test)
    def get_label_columns(self, df) -> list[str]       # helper for UI column picker
```

Key difference from `app/utils.py:preprocess()`: add **StandardScaler normalization** — the raw COVID data has features on wildly different scales which will distort PCA and distance metrics if not normalized. `app/utils.py` currently skips this.

### Ryan — `embeddings/autoencoder.py` (unblocked)

Restore the `Autoencoder` class deleted in commit `b21c3ea`. `embeddings/pipeline.py` imports it directly — until it's restored, the `"autoencoder"` method of `EmbeddingPipeline` will crash on import.

Minimum interface needed by `pipeline.py`:
```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims)
    def forward(self, x) -> (x_hat, z)   # reconstruction + latent vector
    def encode(self, x) -> z             # used by EmbeddingPipeline.transform()
```

### Stella — `app/main.py` pipeline wiring (blocked on Ryan + Ethan)

`app/main.py` currently has a tab scaffold but passes a raw DataFrame to the viz components; no ML runs. Once `dataset_processor.py` and `autoencoder.py` are done, wire the full pipeline inside `app/main.py` using `st.session_state` to cache results across rerenders:

```
upload → label column picker → DatasetProcessor → EmbeddingPipeline.fit()
    → build_classifier() + train → DriftConfig UI → simulate_batches()
    → for each batch: transform + evaluate_batch() + cluster_tracker + neighbor_instability
    → compute_drift_score() per batch → pass results to viz components
```

### Stella — viz components (blocked on pipeline wiring)

Each component receives pre-computed results (not raw data), so they stay pure visualization. Suggested approach with Plotly:

- `embedding_viz.py`: 2D PCA scatter (project to 2D if `n_components > 2`); color by batch index; animate over time with a slider
- `cluster_viz.py`: bar chart of per-cluster centroid shift distances; line chart of reassignment rate over batches
- `retrieval_viz.py`: line chart of neighbor instability score per batch
- `performance_viz.py`: multi-line chart of accuracy/F1/ROC-AUC; calibration curves per batch (Plotly subplots)

### Final integration checklist

- [ ] `data/dataset_processor.py` — DatasetProcessor with StandardScaler normalization (Ethan)
- [ ] `embeddings/autoencoder.py` — Autoencoder class restored (Ryan)
- [ ] `app/main.py` — full pipeline wired end-to-end (Stella, after above)
- [ ] 4 viz components implemented (Stella, after pipeline wiring)
- [ ] Place `data/raw/covid19.csv` locally and smoke-test the full pipeline
- [ ] Add `data/raw/covid19.csv` path note to `.env.example`

## Dataset

Primary: COVID-19 tabular dataset from Kaggle (`data/raw/covid19.csv`). The `data/` directory is gitignored — place datasets there locally.
