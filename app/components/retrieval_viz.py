"""Nearest-neighbor retrieval stability visualization."""

import streamlit as st
import plotly.graph_objects as go


def render_retrieval_viz(batch_results):
    st.subheader("Retrieval Stability (Neighbor Instability)")

    batch_indices = [r.index for r in batch_results]
    nb_scores = [r.neighbor_instability for r in batch_results]
    cosine_shifts = [r.drift_score.mean_cosine_shift for r in batch_results]
    euclidean_shifts = [r.drift_score.mean_euclidean_shift for r in batch_results]

    # --- Neighbor instability line chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=batch_indices, y=nb_scores,
        mode="lines+markers",
        line=dict(color="mediumpurple", width=2),
        marker=dict(size=7),
        name="Neighbor instability",
    ))
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="red", opacity=0.05,
                  annotation_text="High instability", annotation_position="top left")
    fig.update_layout(
        title="k-NN Instability Score Over Batches",
        xaxis_title="Batch", yaxis_title="Instability [0–1]",
        yaxis=dict(range=[0, 1]),
        height=320, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Distance shift breakdown ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=batch_indices, y=cosine_shifts,
        mode="lines+markers", name="Mean cosine shift",
        line=dict(color="steelblue", width=2), marker=dict(size=6),
    ))
    fig2.add_trace(go.Scatter(
        x=batch_indices, y=euclidean_shifts,
        mode="lines+markers", name="Mean Euclidean shift",
        line=dict(color="seagreen", width=2, dash="dot"), marker=dict(size=6),
    ))
    fig2.update_layout(
        title="Embedding Distance Shift Over Batches",
        xaxis_title="Batch", yaxis_title="Distance",
        height=280, margin=dict(t=40, b=20),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Summary table ---
    st.caption("Per-batch retrieval summary")
    import pandas as pd
    table = pd.DataFrame({
        "Batch": batch_indices,
        "Neighbor instability": [f"{v:.3f}" for v in nb_scores],
        "Mean cosine shift": [f"{v:.3f}" for v in cosine_shifts],
        "Mean Euclidean shift": [f"{v:.3f}" for v in euclidean_shifts],
    })
    st.dataframe(table, use_container_width=True, hide_index=True)
