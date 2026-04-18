"""Accuracy, ROC-AUC, F1, and calibration visualizations over drift batches."""

import streamlit as st


def render_performance_viz(dataset, embedding_method: str, n_components: int):
    """Placeholder: per-batch classification performance trends."""
    st.subheader("Model Performance Over Time")
    st.info("Connect PerformanceTracker to populate accuracy, ROC-AUC, and F1 trend lines.")
    # TODO: line charts for accuracy/F1/ROC-AUC; calibration curve per batch
