"""2D projection of embedding space across drift batches."""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from embeddings.pca import PCA


def render_embedding_viz(batch_results, baseline_emb: np.ndarray):
    st.subheader("Embedding Space (2D Projection)")

    # Project all embeddings to 2D using PCA fit on baseline
    pca2 = PCA(n_components=2)
    baseline_2d = pca2.fit_transform(baseline_emb)

    batch_indices = [r.index for r in batch_results]
    selected = st.select_slider("Batch", options=batch_indices,
                                value=batch_results[-1].index,
                                format_func=lambda i: f"Batch {i}")

    result = batch_results[selected]
    drift_2d = pca2.transform(result.emb)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=baseline_2d[:, 0], y=baseline_2d[:, 1],
        mode="markers",
        marker=dict(size=4, color="steelblue", opacity=0.5),
        name="Baseline",
    ))
    fig.add_trace(go.Scatter(
        x=drift_2d[:, 0], y=drift_2d[:, 1],
        mode="markers",
        marker=dict(size=4, color="tomato", opacity=0.5),
        name=f"Batch {selected} (drifted)",
    ))
    fig.update_layout(
        xaxis_title="PC 1", yaxis_title="PC 2",
        legend=dict(orientation="h"),
        height=480,
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drift score trend mini-chart
    st.caption("Drift score over all batches")
    scores = [r.drift_score.score for r in batch_results]
    st.line_chart({"Drift score": scores})
