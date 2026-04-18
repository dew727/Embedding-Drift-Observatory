"""Cluster centroid shift and membership change visualizations."""

import streamlit as st


def render_cluster_viz(dataset, embedding_method: str, n_components: int):
    """Placeholder: centroid movement and cluster reassignment rate over batches."""
    st.subheader("Cluster Dynamics")
    st.info("Connect ClusterTracker to populate centroid shift and reassignment rate charts.")
    # TODO: bar chart of per-cluster centroid shift distances, line chart of reassignment rate
