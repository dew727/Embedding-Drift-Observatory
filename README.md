# 🔭 Embedding Drift Observatory

> **Representation Stability Under Distribution Shift** — A web platform for monitoring how learned embeddings evolve as medical data distributions change over time.

---

## Overview

Machine learning models often degrade after deployment due to **distribution shift** — but most monitoring tools only track surface-level metrics like accuracy, without understanding *why* performance is failing.

The **Embedding Drift Observatory** goes deeper. It analyzes how learned embeddings (the internal vector representations models rely on) change as data distributions shift. By studying the geometry of the embedding space — including cluster structure, similarity relationships, and downstream task performance — this system identifies deeper signals of model instability before or alongside performance decay.

**Key question:** *Are learned embeddings stable under distribution shift?*

---

## Features

- **Upload any dataset** and choose an embedding method (PCA or Autoencoder)
- **Simulate real-world deployment** with controlled distribution shift and noise injection
- **Interactive dashboard** tracking embedding movement, cluster dynamics, retrieval behavior, and model performance over time
- **Instability score** that recommends whether your model needs retraining
- **Modality-agnostic** — works on any dataset once embedded into vector space

---

## ML Components

### Representation Learning
| Method | Details |
|---|---|
| PCA | Implemented from scratch (mean-centering, covariance matrix, eigendecomposition/SVD) |
| Autoencoder | PyTorch-based encoder-decoder architecture |
| k-Means | Optional: for cluster structure discovery |

### Embedding Space Analysis
- Cosine & Euclidean distance shifts
- Centroid movement in latent space
- Cluster reassignment rates
- Nearest neighbor instability

### Downstream Tasks
- **Classification** — MLP or logistic regression
- **Clustering** — k-means
- **Retrieval** — nearest neighbor search

### Drift Simulation
- **Covariate shift** — feature distribution changes
- **Prior shift** — class imbalance changes
- **Concept drift** — label relationship changes
- **Noise/corruption injection** — missingness, Gaussian noise

### Monitoring Metrics
- ROC-AUC, F1 score, accuracy over time
- Calibration curves
- Embedding distribution divergence
- Retrieval consistency metrics

---

## Project Pipeline

```
Data Ingestion
    ↓
Preprocessing (missing values, encoding, normalization)
    ↓
Data Splitting (baseline train/test + future batches)
    ↓
Embedding Learning (PCA / Autoencoder on baseline)
    ↓
Baseline Modeling (classifier, clustering, retrieval)
    ↓
Drift Simulation (covariate / prior / concept / noise)
    ↓
Re-Embedding (transform drifted batches)
    ↓
Embedding Analysis (distance, cluster, neighborhood metrics)
    ↓
Performance Evaluation (per-batch metric tracking)
    ↓
Drift Quantification (instability score)
    ↓
Visualization (interactive dashboard)
```

---


## Dataset

**Primary:** [COVID-19 Dataset](https://www.kaggle.com/) — used to simulate evolving patient distributions and demonstrate real-world deployment conditions.

The platform is designed to be **modality-agnostic** — any tabular dataset with numeric features can be uploaded and analyzed.

---

## Team

| Member | Role |
|---|---|
| **Ryan** | Representation Learning Lead — PCA from scratch, Autoencoder (PyTorch), embedding pipeline |
| **Ethan Xu** | Drift Simulation & Data Pipeline — drift generators, batch/time simulation, preprocessing |
| **Quentin Liang** | Embedding Analysis (Geometry) — distance metrics, cluster tracking, embedding space changes |
| **Terrence Lin** | Downstream Tasks & Evaluation — classifiers, ROC-AUC/F1/calibration, per-batch tracking |
| **Stella** | Frontend & Integration — Streamlit dashboard, visualizations, backend integration |


**Final Presentation:** April 27 – May 1
