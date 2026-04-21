"""Dataset upload sidebar component."""

import streamlit as st
from app.utils import load_csv
from data.dataset_processor import DatasetProcessor


def render_upload() -> tuple:
    """Render file upload and label column picker in the sidebar.

    Returns:
        (df, label_col) — df is None until a file is uploaded.
    """
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    df = None
    label_col = None

    if uploaded is not None:
        df = load_csv(uploaded)
        st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
        with st.expander("Preview"):
            st.dataframe(df.head())

        proc = DatasetProcessor()
        candidates = proc.get_label_columns(df)
        if not candidates:
            st.error("No suitable label column found (need a column with <50 unique values).")
            return None, None

        label_col = st.selectbox("Label column", candidates)

    return df, label_col
