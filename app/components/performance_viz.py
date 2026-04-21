"""Classification performance and drift score visualizations over time."""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_performance_viz(batch_results, perf_tracker):
    st.subheader("Model Performance Over Time")

    batch_indices = [r.index for r in batch_results]
    accuracies = [r.metrics.accuracy for r in batch_results]
    f1_scores = [r.metrics.f1 for r in batch_results]
    roc_aucs = [r.metrics.roc_auc for r in batch_results]
    drift_scores = [r.drift_score.score for r in batch_results]

    # --- Classification metrics + drift score (dual y-axis) ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for name, values, color in [
        ("Accuracy", accuracies, "steelblue"),
        ("F1 (weighted)", f1_scores, "seagreen"),
        ("ROC-AUC", roc_aucs, "darkorange"),
    ]:
        valid = [(i, v) for i, v in zip(batch_indices, values) if not np.isnan(v)]
        if valid:
            xs, ys = zip(*valid)
            fig.add_trace(
                go.Scatter(x=list(xs), y=list(ys), mode="lines+markers",
                           name=name, line=dict(color=color, width=2),
                           marker=dict(size=6)),
                secondary_y=False,
            )

    fig.add_trace(
        go.Scatter(x=batch_indices, y=drift_scores,
                   mode="lines+markers", name="Drift score",
                   line=dict(color="red", width=2, dash="dash"),
                   marker=dict(size=6, symbol="diamond")),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Classification metric", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text="Drift score", range=[0, 1], secondary_y=True)
    fig.update_layout(
        title="Performance Metrics vs. Drift Score",
        xaxis_title="Batch",
        height=380, margin=dict(t=40, b=20),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Calibration curves for a selected batch ---
    st.caption("Calibration curve — select a batch")
    valid_batches = [r for r in batch_results if r.metrics.fraction_of_positives is not None]
    if valid_batches:
        selected = st.select_slider(
            "Batch (calibration)", options=[r.index for r in valid_batches],
            value=valid_batches[-1].index, key="perf_cal_slider",
        )
        result = next(r for r in valid_batches if r.index == selected)
        frac_pos = result.metrics.fraction_of_positives
        mean_pred = result.metrics.mean_predicted_value

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dot"), name="Perfect calibration",
        ))
        fig_cal.add_trace(go.Scatter(
            x=list(mean_pred), y=list(frac_pos),
            mode="lines+markers", name=f"Batch {selected}",
            line=dict(color="darkorange", width=2), marker=dict(size=8),
        ))
        fig_cal.update_layout(
            xaxis_title="Mean predicted probability",
            yaxis_title="Fraction of positives",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
            height=300, margin=dict(t=10, b=20),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    # --- Summary stats ---
    summary = perf_tracker.summary()
    if summary:
        st.caption("Aggregate performance summary across all batches")
        cols = st.columns(3)
        for col, (metric, label) in zip(cols, [
            ("accuracy", "Accuracy"), ("f1", "F1"), ("roc_auc", "ROC-AUC")
        ]):
            mean = summary[metric]["mean"]
            std = summary[metric]["std"]
            col.metric(label, f"{mean:.3f}", delta=f"±{std:.3f}", delta_color="off")
