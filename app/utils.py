"""Shared UI helpers for the Streamlit dashboard."""

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV file into a DataFrame (cached by file content)."""
    return pd.read_csv(uploaded_file)


def preprocess(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Split a DataFrame into numeric feature matrix X and label vector y.

    Drops non-numeric columns except the label column.
    Returns X (n_samples, n_features), y (n_samples,), and feature names.
    """
    y = df[label_col].values
    feature_df = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    X = feature_df.values.astype(np.float32)
    # Fill any remaining NaNs with column means
    col_means = np.nanmean(X, axis=0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(col_means, nan_idx[1])
    return X, y, list(feature_df.columns)


def train_test_split_indices(n: int, test_ratio: float = 0.2, seed: int = 42):
    """Return (train_idx, test_idx) arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    return idx[:split], idx[split:]
