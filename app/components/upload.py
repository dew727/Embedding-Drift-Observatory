"""Dataset upload and pipeline configuration sidebar component."""

import streamlit as st
import pandas as pd
from app.utils import load_csv


def render_upload():
    """Render file upload and pipeline settings in the sidebar.

    Returns:
        (df, embedding_method, n_components) or (None, ..., ...) if no file uploaded.
    """
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    df = None
    if uploaded is not None:
        df = load_csv(uploaded)
        st.success(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
        with st.expander("Preview"):
            st.dataframe(df.head())

    embedding_method = st.selectbox("Embedding method", ["pca", "autoencoder"])
    n_components = st.slider("Embedding dimensions", min_value=2, max_value=64, value=16)

    return df, embedding_method, n_components
