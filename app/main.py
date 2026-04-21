"""Streamlit entry point for the Embedding Drift Observatory.

Run with:
    streamlit run app/main.py
"""

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Embedding Drift Observatory",
    page_icon="🔭",
    layout="wide",
)

from app.components.upload import render_upload
from app.components.embedding_viz import render_embedding_viz
from app.components.cluster_viz import render_cluster_viz
from app.components.retrieval_viz import render_retrieval_viz
from app.components.performance_viz import render_performance_viz
from app.utils import BatchResult

from data.dataset_processor import DatasetProcessor
from embeddings.pipeline import EmbeddingPipeline
from drift.batch_simulator import DriftConfig, simulate_batches
from analysis.cluster_tracker import ClusterTracker
from analysis.distance_metrics import mean_pairwise_shift
from analysis.neighbor_instability import neighbor_instability as compute_neighbor_instability
from analysis.drift_score import compute_drift_score
from evaluation.classifier import build_classifier, train_classifier
from evaluation.metrics import evaluate_batch
from evaluation.performance_tracker import PerformanceTracker


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔭 Observatory")

    st.header("1 · Dataset")
    df, label_col = render_upload()

    st.header("2 · Embedding")
    embed_method = st.selectbox("Method", ["pca", "autoencoder"])
    n_components = st.slider("Dimensions", min_value=2, max_value=64, value=16)

    st.header("3 · Classifier")
    clf_kind = st.selectbox("Classifier", ["logistic", "mlp"])

    st.header("4 · Drift Simulation")
    n_batches = st.slider("Number of batches", min_value=2, max_value=20, value=8)

    with st.expander("Drift settings (applied uniformly to all batches)"):
        covariate_strength = st.slider("Covariate shift strength", 0.0, 3.0, 1.0, 0.1)
        enable_prior = st.checkbox("Prior shift", value=False)
        prior_ratio = st.slider("Prior ratio (majority class)", 0.5, 0.99, 0.8, 0.01) if enable_prior else None
        concept_flip = st.slider("Concept drift (label flip rate)", 0.0, 0.5, 0.0, 0.05)
        noise_std = st.slider("Gaussian noise std", 0.0, 1.0, 0.1, 0.05)
        missing_rate = st.slider("Missingness rate", 0.0, 0.5, 0.0, 0.05)

    n_clusters = st.slider("k-means clusters (analysis)", min_value=2, max_value=16, value=6)

    st.divider()
    run_clicked = st.button("▶ Run Pipeline", type="primary", disabled=(df is None))


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _run_pipeline(df, label_col, embed_method, n_components, clf_kind,
                  n_batches, drift_cfg, n_clusters):
    proc = DatasetProcessor()
    X, y, _ = proc.preprocess(df, label_col)
    X_train, X_test, y_train, y_test = proc.split(X, y)

    # Fit embedder on training data only — frozen for all subsequent transforms
    pipeline = EmbeddingPipeline(method=embed_method, n_components=n_components)
    with st.spinner("Fitting embedder on baseline data…"):
        pipeline.fit(X_train)

    baseline_emb = pipeline.transform(X_train)

    # Fit cluster tracker on baseline embeddings
    cluster_tracker = ClusterTracker(n_clusters=n_clusters)
    cluster_tracker.fit(baseline_emb)

    # Train classifier on baseline embeddings
    clf = build_classifier(clf_kind)
    with st.spinner("Training baseline classifier…"):
        train_classifier(clf, baseline_emb, y_train)

    # Generate drift batches from full dataset
    drift_configs = [drift_cfg] * n_batches
    batch_size = max(50, len(X) // n_batches)
    batches = simulate_batches(X, y, n_batches, drift_configs, batch_size=batch_size)

    perf_tracker = PerformanceTracker()
    batch_results: list[BatchResult] = []

    progress = st.progress(0, text="Running drift batches…")
    for i, batch in enumerate(batches):
        # Embed the drifted batch
        drift_emb = pipeline.transform(batch.X)

        # Embed the same samples WITHOUT drift for comparison metrics
        start = (i * batch_size) % len(X)
        orig_idx = np.arange(start, start + batch_size) % len(X)
        orig_emb = pipeline.transform(X[orig_idx])

        # Geometric analysis
        centroid_shifts = cluster_tracker.centroid_shift(drift_emb)
        reassign_rate = cluster_tracker.reassignment_rate(orig_emb, drift_emb)
        nb_instability = compute_neighbor_instability(orig_emb, drift_emb, k=5)
        mean_cos = mean_pairwise_shift(orig_emb, drift_emb, metric="cosine")
        mean_euc = mean_pairwise_shift(orig_emb, drift_emb, metric="euclidean")
        drift_score = compute_drift_score(
            mean_cos, mean_euc, nb_instability,
            float(centroid_shifts.mean()), reassign_rate,
        )

        # Performance evaluation — skip batch if only one class present
        try:
            metrics = evaluate_batch(clf, drift_emb, batch.y, batch_index=i)
        except Exception:
            from evaluation.metrics import BatchMetrics
            metrics = BatchMetrics(batch_index=i, accuracy=float("nan"),
                                   f1=float("nan"), roc_auc=float("nan"))
        perf_tracker.log(metrics)

        batch_results.append(BatchResult(
            index=i,
            emb=drift_emb,
            baseline_emb=orig_emb,
            metrics=metrics,
            centroid_shifts=centroid_shifts,
            reassignment_rate=reassign_rate,
            neighbor_instability=nb_instability,
            drift_score=drift_score,
            config=batch.config,
        ))
        progress.progress((i + 1) / n_batches, text=f"Batch {i + 1}/{n_batches} done")

    progress.empty()
    return baseline_emb, batch_results, perf_tracker


if run_clicked and df is not None:
    drift_cfg = DriftConfig(
        covariate_strength=covariate_strength,
        prior_ratio=prior_ratio,
        concept_flip_rate=concept_flip,
        noise_std=noise_std,
        missing_rate=missing_rate,
    )
    with st.spinner("Running pipeline…"):
        baseline_emb, batch_results, perf_tracker = _run_pipeline(
            df, label_col, embed_method, n_components, clf_kind,
            n_batches, drift_cfg, n_clusters,
        )
    st.session_state["results"] = (baseline_emb, batch_results, perf_tracker)
    st.success("Pipeline complete.")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

st.title("🔭 Embedding Drift Observatory")
st.caption("Representation Stability Under Distribution Shift")

if "results" not in st.session_state:
    st.info("Configure the pipeline in the sidebar and click **▶ Run Pipeline** to begin.")
    st.stop()

baseline_emb, batch_results, perf_tracker = st.session_state["results"]

# Aggregate drift score banner
scores = [r.drift_score.score for r in batch_results]
final_score = scores[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Final Drift Score", f"{final_score:.3f}", help="Composite geometric instability score [0–1]")
col2.metric("Peak Drift Score", f"{max(scores):.3f}")
summary = perf_tracker.summary()
if summary:
    col3.metric("Mean ROC-AUC", f"{summary['roc_auc']['mean']:.3f}",
                delta=f"±{summary['roc_auc']['std']:.3f}", delta_color="off")

st.divider()

tab_emb, tab_cluster, tab_retrieval, tab_perf = st.tabs([
    "Embedding Space", "Cluster Dynamics", "Retrieval Stability", "Performance",
])

with tab_emb:
    render_embedding_viz(batch_results, baseline_emb)

with tab_cluster:
    render_cluster_viz(batch_results)

with tab_retrieval:
    render_retrieval_viz(batch_results)

with tab_perf:
    render_performance_viz(batch_results, perf_tracker)
