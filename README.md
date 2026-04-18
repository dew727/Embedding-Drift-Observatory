# 🔭 Embedding Drift Observatory

> **Representation Stability Under Distribution Shift** — A web platform for monitoring how learned embeddings evolve as medical data distributions change over time.

---

## Overview

Machine learning models often degrade after deployment due to **distribution shift** — but most monitoring tools only track surface-level metrics like accuracy, without understanding *why* performance is failing.

The **Embedding Drift Observatory** goes deeper. It analyzes how learned embeddings (the internal vector representations models rely on) change as data distributions shift. By studying the geometry of the embedding space — including cluster structure, similarity relationships, and downstream task performance — this system identifies deeper signals of model instability before or alongside performance decay.

**Key question:** *Are learned embeddings stable under distribution shift?*

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Then upload any numeric CSV dataset from the sidebar to begin.

---

## Project Structure

```
embedding-drift-observatory/
│
├── app/                        # Streamlit dashboard
│   ├── main.py                 # Entry point (streamlit run app/main.py)
│   ├── utils.py                # Shared helpers: CSV loading, preprocessing, splitting
│   └── components/             # One file per dashboard tab/view
│
├── embeddings/                 # Representation learning
│   ├── pca.py                  # PCA from scratch (SVD-based)
│   ├── autoencoder.py          # PyTorch encoder-decoder
│   └── pipeline.py             # Unified fit/transform interface
│
├── drift/                      # Drift simulation
│   ├── covariate_shift.py      # Feature distribution perturbation
│   ├── prior_shift.py          # Class imbalance resampling
│   ├── concept_drift.py        # Label flip injection
│   ├── noise_injection.py      # Gaussian noise & missingness
│   └── batch_simulator.py      # Composable per-batch drift via DriftConfig
│
├── analysis/                   # Embedding geometry metrics
│   ├── distance_metrics.py     # Cosine & Euclidean shift
│   ├── cluster_tracker.py      # k-means centroid shift & reassignment rate
│   ├── neighbor_instability.py # k-NN stability metric
│   └── drift_score.py          # Weighted composite instability score
│
├── evaluation/                 # Downstream task evaluation
│   ├── classifier.py           # Logistic regression / MLP
│   ├── metrics.py              # ROC-AUC, F1, accuracy, calibration (per batch)
│   └── performance_tracker.py  # Temporal metric accumulator
│
├── data/
│   ├── raw/                    # Place datasets here (gitignored)
│   └── processed/              # Preprocessed outputs (gitignored)
│
├── notebooks/
│   └── exploration.ipynb       # EDA and prototyping
│
├── requirements.txt
└── .env.example
```

---

## Features

- **Upload any dataset** and choose an embedding method (PCA or Autoencoder)
- **Simulate real-world deployment** with controlled distribution shift and noise injection
- **Interactive dashboard** tracking embedding movement, cluster dynamics, retrieval behavior, and model performance over time
- **Instability score** that recommends whether your model needs retraining
- **Modality-agnostic** — works on any tabular numeric dataset

---

## ML Components

### Representation Learning
| Method | Details |
|---|---|
| PCA | Implemented from scratch (mean-centering, covariance matrix, SVD) |
| Autoencoder | PyTorch-based encoder-decoder; encoder output is the embedding |

### Drift Simulation
| Type | File | Description |
|---|---|---|
| Covariate shift | `drift/covariate_shift.py` | Shifts feature means by N standard deviations |
| Prior shift | `drift/prior_shift.py` | Resamples to change class proportions |
| Concept drift | `drift/concept_drift.py` | Randomly flips a fraction of labels |
| Noise injection | `drift/noise_injection.py` | Gaussian noise or random missingness |

Drift types are composed per batch via `DriftConfig` in `drift/batch_simulator.py`.

### Embedding Space Analysis
- Cosine & Euclidean distance shifts (`analysis/distance_metrics.py`)
- Centroid movement and cluster reassignment rate (`analysis/cluster_tracker.py`)
- Nearest-neighbor instability (`analysis/neighbor_instability.py`)
- Aggregate drift score (`analysis/drift_score.py`)

### Downstream Evaluation
- Classification: Logistic Regression or MLP (`evaluation/classifier.py`)
- Per-batch metrics: ROC-AUC, F1, accuracy, calibration (`evaluation/metrics.py`)
- Temporal trend tracking (`evaluation/performance_tracker.py`)

---

## Pipeline

```
Data Ingestion → Preprocessing → Baseline Split
    → Embedding Learning (fit once on baseline)
    → Baseline Modeling
    → Drift Simulation (per batch)
    → Re-Embedding (transform with frozen embedder)
    → Embedding Analysis + Performance Evaluation
    → Drift Score → Dashboard
```

---

## Dataset

**Primary:** COVID-19 tabular dataset from Kaggle — used to simulate evolving patient distributions.

Place datasets in `data/raw/` (gitignored). The platform accepts any numeric CSV.

---

## Team

| Member | Role |
|---|---|
| **Ryan** | Representation Learning — `embeddings/` |
| **Ethan Xu** | Drift Simulation & Data Pipeline — `drift/` |
| **Quentin Liang** | Embedding Analysis — `analysis/` |
| **Terrence Lin** | Downstream Tasks & Evaluation — `evaluation/` |
| **Stella** | Frontend & Integration — `app/` |

**Final Presentation:** April 27 – May 1
