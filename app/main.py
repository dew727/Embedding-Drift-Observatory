"""Streamlit entry point for the Embedding Drift Observatory dashboard.

Run with:
    streamlit run app/main.py
"""

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

st.title("🔭 Embedding Drift Observatory")
st.caption("Representation Stability Under Distribution Shift")

# Sidebar: dataset upload and pipeline configuration
with st.sidebar:
    st.header("Configuration")
    dataset, embedding_method, n_components = render_upload()

if dataset is None:
    st.info("Upload a dataset in the sidebar to begin.")
    st.stop()

# Tabs for each monitoring view
tab_emb, tab_cluster, tab_retrieval, tab_perf = st.tabs([
    "Embedding Space", "Cluster Dynamics", "Retrieval Behavior", "Performance"
])

with tab_emb:
    render_embedding_viz(dataset, embedding_method, n_components)

with tab_cluster:
    render_cluster_viz(dataset, embedding_method, n_components)

with tab_retrieval:
    render_retrieval_viz(dataset, embedding_method, n_components)

with tab_perf:
    render_performance_viz(dataset, embedding_method, n_components)
