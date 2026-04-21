"""Cluster centroid shift and reassignment rate visualizations."""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_cluster_viz(batch_results):
    st.subheader("Cluster Dynamics")

    n_clusters = len(batch_results[0].centroid_shifts)
    batch_indices = [r.index for r in batch_results]

    # --- Centroid shift heatmap (batches × clusters) ---
    shift_matrix = np.array([r.centroid_shifts for r in batch_results])  # (n_batches, n_clusters)

    fig_heat = go.Figure(go.Heatmap(
        z=shift_matrix,
        x=[f"C{k}" for k in range(n_clusters)],
        y=[f"Batch {i}" for i in batch_indices],
        colorscale="Reds",
        colorbar=dict(title="Shift dist."),
    ))
    fig_heat.update_layout(
        title="Centroid Shift Distance per Cluster × Batch",
        xaxis_title="Cluster", yaxis_title="Batch",
        height=320, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Reassignment rate over time ---
    reassign = [r.reassignment_rate for r in batch_results]
    fig_line = go.Figure(go.Scatter(
        x=batch_indices, y=reassign,
        mode="lines+markers",
        line=dict(color="darkorange", width=2),
        marker=dict(size=7),
        name="Reassignment rate",
    ))
    fig_line.update_layout(
        title="Cluster Reassignment Rate Over Batches",
        xaxis_title="Batch", yaxis_title="Fraction reassigned",
        yaxis=dict(range=[0, 1]),
        height=280, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Per-batch centroid shift bar chart (slider) ---
    st.caption("Per-cluster centroid shift — select a batch")
    selected = st.select_slider("Batch (centroid detail)", options=batch_indices,
                                value=batch_results[-1].index,
                                key="cluster_batch_slider")
    shifts = batch_results[selected].centroid_shifts
    fig_bar = go.Figure(go.Bar(
        x=[f"C{k}" for k in range(n_clusters)],
        y=shifts,
        marker_color="salmon",
    ))
    fig_bar.update_layout(
        xaxis_title="Cluster", yaxis_title="Distance from baseline centroid",
        height=260, margin=dict(t=10, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
