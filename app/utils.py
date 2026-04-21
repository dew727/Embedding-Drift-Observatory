"""Shared helpers for the Streamlit dashboard."""

import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass

from analysis.drift_score import DriftScoreResult
from evaluation.metrics import BatchMetrics
from drift.batch_simulator import DriftConfig


@dataclass
class BatchResult:
    """All per-batch outputs produced by the pipeline loop."""
    index: int
    emb: np.ndarray              # drifted batch embeddings (n_samples, latent_dim)
    baseline_emb: np.ndarray     # same samples embedded without drift
    metrics: BatchMetrics
    centroid_shifts: np.ndarray  # per-cluster (n_clusters,)
    reassignment_rate: float
    neighbor_instability: float
    drift_score: DriftScoreResult
    config: DriftConfig


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def preprocess(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Numeric-only preprocessing without normalization (used as fallback).

    Prefer DatasetProcessor.preprocess() for the full pipeline — it adds StandardScaler.
    """
    y = df[label_col].values
    feature_df = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    X = feature_df.values.astype(np.float32)
    col_means = np.nanmean(X, axis=0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(col_means, nan_idx[1])
    return X, y, list(feature_df.columns)


def train_test_split_indices(n: int, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    return idx[:split], idx[split:]
