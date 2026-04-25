import { useMemo, useState } from "react"
import Plot from "react-plotly.js"
import "./App.css"

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  "https://backend-api-production-7b9a.up.railway.app"
const TABS = [
  { id: "embedding", label: "Embedding Space" },
  { id: "clusters", label: "Cluster Dynamics" },
  { id: "retrieval", label: "Retrieval Stability" },
  { id: "performance", label: "Performance" },
]

const INITIAL_FORM = {
  embed_method: "pca",
  n_components: 8,
  clf_kind: "logistic",
  n_batches: 6,
  n_clusters: 4,
  covariate_strength: 1,
  noise_std: 0.1,
  concept_flip_rate: 0,
  prior_ratio: "",
  missing_rate: 0,
}

function formatMetric(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A"
  }
  return Number(value).toFixed(digits)
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A"
  }
  return `${Math.round(Number(value) * 100)}%`
}

function toneForVerdict(verdict) {
  if (verdict === "retrain") return "danger"
  if (verdict === "monitor") return "warning"
  return "success"
}

function MetricTile({ label, value, caption }) {
  return (
    <article className="metric-tile">
      <span className="metric-tile__label">{label}</span>
      <strong className="metric-tile__value">{value}</strong>
      <span className="metric-tile__caption">{caption}</span>
    </article>
  )
}

function SectionCard({ title, subtitle, children }) {
  return (
    <section className="section-card">
      <div className="section-card__header">
        <h3>{title}</h3>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {children}
    </section>
  )
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadMeta, setUploadMeta] = useState(null)
  const [labelCol, setLabelCol] = useState("")
  const [form, setForm] = useState(INITIAL_FORM)
  const [activeTab, setActiveTab] = useState("embedding")
  const [selectedBatchIndex, setSelectedBatchIndex] = useState(0)
  const [result, setResult] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState("")

  const selectedBatch = result?.batches?.[selectedBatchIndex] ?? null
  const verdictTone = toneForVerdict(result?.verdict?.verdict)

  const summaryTiles = useMemo(() => {
    if (!result?.summary) {
      return []
    }

    return [
      {
        label: "Mean Accuracy",
        value: formatMetric(result.summary.accuracy?.mean),
        caption: `std ${formatMetric(result.summary.accuracy?.std)}`,
      },
      {
        label: "Mean F1",
        value: formatMetric(result.summary.f1?.mean),
        caption: `std ${formatMetric(result.summary.f1?.std)}`,
      },
      {
        label: "Mean ROC-AUC",
        value: formatMetric(result.summary.roc_auc?.mean),
        caption: `std ${formatMetric(result.summary.roc_auc?.std)}`,
      },
      {
        label: "Train / Test",
        value: `${result.n_train} / ${result.n_test}`,
        caption: `${result.feature_names.length} engineered features`,
      },
    ]
  }, [result])

  const timeline = useMemo(() => {
    if (!result?.batches) {
      return {
        indices: [],
        drift: [],
        neighborInstability: [],
        reassignment: [],
        accuracy: [],
        f1: [],
        rocAuc: [],
      }
    }

    return {
      indices: result.batches.map((batch) => batch.index + 1),
      drift: result.batches.map((batch) => batch.drift_score),
      neighborInstability: result.batches.map((batch) => batch.neighbor_instability),
      reassignment: result.batches.map((batch) => batch.reassignment_rate),
      accuracy: result.batches.map((batch) => batch.accuracy),
      f1: result.batches.map((batch) => batch.f1),
      rocAuc: result.batches.map((batch) => batch.roc_auc),
    }
  }, [result])

  async function uploadDataset(file) {
    if (!file) {
      return
    }

    setIsUploading(true)
    setError("")
    setUploadMeta(null)
    setResult(null)

    try {
      const body = new FormData()
      body.append("file", file)

      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body,
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || "Upload failed.")
      }

      const payload = await response.json()
      setUploadMeta(payload)
      setLabelCol(payload.label_candidates[0] || payload.columns[0] || "")
      setSelectedBatchIndex(0)
    } catch (err) {
      setError(err.message || "Unable to upload dataset.")
    } finally {
      setIsUploading(false)
    }
  }

  async function handleRun() {
    if (!uploadMeta?.upload_id || !labelCol) {
      setError("Upload a dataset and pick a label column before running the pipeline.")
      return
    }

    setIsRunning(true)
    setError("")

    try {
      const payload = {
        upload_id: uploadMeta.upload_id,
        label_col: labelCol,
        embed_method: form.embed_method,
        n_components: Number(form.n_components),
        clf_kind: form.clf_kind,
        n_batches: Number(form.n_batches),
        n_clusters: Number(form.n_clusters),
        drift_config: {
          covariate_strength: Number(form.covariate_strength),
          noise_std: Number(form.noise_std),
          concept_flip_rate: Number(form.concept_flip_rate),
          prior_ratio: form.prior_ratio === "" ? null : Number(form.prior_ratio),
          missing_rate: Number(form.missing_rate),
        },
      }

      const response = await fetch(`${API_BASE}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || "Pipeline run failed.")
      }

      const nextResult = await response.json()
      setResult(nextResult)
      setSelectedBatchIndex(0)
      setActiveTab("embedding")
    } catch (err) {
      setError(err.message || "Unable to run the pipeline.")
    } finally {
      setIsRunning(false)
    }
  }

  function updateField(key, value) {
    setForm((current) => ({
      ...current,
      [key]: value,
    }))
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <p className="eyebrow">React Frontend</p>
          <h1>Embedding Drift Observatory</h1>
          <p className="brand-block__lede">
            Upload a tabular dataset, pick the prediction target, and stress the learned
            embedding across drifting batches from the FastAPI pipeline.
          </p>
        </div>

        <SectionCard title="Dataset" subtitle="Step 1: upload a CSV and inspect candidate labels.">
          <label className="upload-card">
            <span className="upload-card__title">CSV upload</span>
            <span className="upload-card__hint">Numeric columns are embedded; low-cardinality columns are suggested as labels.</span>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => {
                const file = event.target.files?.[0] ?? null
                setSelectedFile(file)
              }}
            />
            <strong>{selectedFile ? selectedFile.name : "Choose a CSV file"}</strong>
          </label>

          <button
            className="primary-button"
            type="button"
            onClick={() => uploadDataset(selectedFile)}
            disabled={!selectedFile || isUploading}
          >
            {isUploading ? "Uploading..." : "Upload dataset"}
          </button>

          {uploadMeta ? (
            <div className="dataset-meta">
              <div>
                <span>Rows</span>
                <strong>{uploadMeta.n_rows}</strong>
              </div>
              <div>
                <span>Columns</span>
                <strong>{uploadMeta.n_cols}</strong>
              </div>
              <div>
                <span>Upload ID</span>
                <strong>{uploadMeta.upload_id.slice(0, 8)}...</strong>
              </div>
            </div>
          ) : null}

          <label className="field">
            <span>Label column</span>
            <select value={labelCol} onChange={(event) => setLabelCol(event.target.value)} disabled={!uploadMeta}>
              {!uploadMeta ? <option value="">Upload first</option> : null}
              {uploadMeta?.label_candidates?.length
                ? uploadMeta.label_candidates.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))
                : uploadMeta?.columns?.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
            </select>
          </label>
        </SectionCard>

        <SectionCard title="Pipeline" subtitle="Step 2: choose the embedding and classifier settings.">
          <div className="field-grid">
            <label className="field">
              <span>Embedding method</span>
              <select value={form.embed_method} onChange={(event) => updateField("embed_method", event.target.value)}>
                <option value="pca">PCA</option>
                <option value="autoencoder">Autoencoder</option>
              </select>
            </label>

            <label className="field">
              <span>Classifier</span>
              <select value={form.clf_kind} onChange={(event) => updateField("clf_kind", event.target.value)}>
                <option value="logistic">Logistic Regression</option>
                <option value="mlp">MLP</option>
              </select>
            </label>
          </div>

          <label className="range-field">
            <div>
              <span>Embedding dimensions</span>
              <strong>{form.n_components}</strong>
            </div>
            <input
              type="range"
              min="2"
              max="64"
              step="1"
              value={form.n_components}
              onChange={(event) => updateField("n_components", Number(event.target.value))}
            />
          </label>

          <label className="range-field">
            <div>
              <span>Temporal batches</span>
              <strong>{form.n_batches}</strong>
            </div>
            <input
              type="range"
              min="2"
              max="20"
              step="1"
              value={form.n_batches}
              onChange={(event) => updateField("n_batches", Number(event.target.value))}
            />
          </label>

          <label className="range-field">
            <div>
              <span>Cluster count</span>
              <strong>{form.n_clusters}</strong>
            </div>
            <input
              type="range"
              min="2"
              max="16"
              step="1"
              value={form.n_clusters}
              onChange={(event) => updateField("n_clusters", Number(event.target.value))}
            />
          </label>
        </SectionCard>

        <SectionCard title="Drift Settings" subtitle="Step 3: control the simulated deployment shift.">
          <label className="range-field">
            <div>
              <span>Covariate strength</span>
              <strong>{formatMetric(form.covariate_strength, 2)}</strong>
            </div>
            <input
              type="range"
              min="0"
              max="3"
              step="0.05"
              value={form.covariate_strength}
              onChange={(event) => updateField("covariate_strength", Number(event.target.value))}
            />
          </label>

          <label className="range-field">
            <div>
              <span>Noise std</span>
              <strong>{formatMetric(form.noise_std, 2)}</strong>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.02"
              value={form.noise_std}
              onChange={(event) => updateField("noise_std", Number(event.target.value))}
            />
          </label>

          <label className="range-field">
            <div>
              <span>Concept flip rate</span>
              <strong>{formatMetric(form.concept_flip_rate, 2)}</strong>
            </div>
            <input
              type="range"
              min="0"
              max="0.5"
              step="0.01"
              value={form.concept_flip_rate}
              onChange={(event) => updateField("concept_flip_rate", Number(event.target.value))}
            />
          </label>

          <label className="range-field">
            <div>
              <span>Missing rate</span>
              <strong>{formatMetric(form.missing_rate, 2)}</strong>
            </div>
            <input
              type="range"
              min="0"
              max="0.5"
              step="0.01"
              value={form.missing_rate}
              onChange={(event) => updateField("missing_rate", Number(event.target.value))}
            />
          </label>

          <label className="field">
            <span>Prior ratio</span>
            <input
              type="number"
              min="0.5"
              max="0.99"
              step="0.01"
              placeholder="Disabled"
              value={form.prior_ratio}
              onChange={(event) => updateField("prior_ratio", event.target.value)}
            />
          </label>

          <button
            className="primary-button primary-button--run"
            type="button"
            onClick={handleRun}
            disabled={!uploadMeta || !labelCol || isRunning}
          >
            {isRunning ? "Running pipeline..." : "Run observatory"}
          </button>
        </SectionCard>

        {error ? <div className="error-banner">{error}</div> : null}
      </aside>

      <section className="main-panel">
        <header className={`verdict-banner verdict-banner--${verdictTone}`}>
          <div>
            <p className="eyebrow">Pipeline verdict</p>
            <h2>{result?.verdict?.label || "Ready for a pipeline run"}</h2>
            <p className="verdict-banner__reason">
              {result?.verdict?.reason ||
                "Once you run the pipeline, this banner will summarize whether the embedding looks stable, worth monitoring, or ready for retraining."}
            </p>
          </div>

          <div className="verdict-signals">
            <div>
              <span>Peak drift</span>
              <strong>{formatMetric(result?.verdict?.signals?.peak_drift_score)}</strong>
            </div>
            <div>
              <span>Neighbor instability</span>
              <strong>{formatMetric(result?.verdict?.signals?.max_neighbor_instability)}</strong>
            </div>
            <div>
              <span>Accuracy drop</span>
              <strong>{formatPercent(result?.verdict?.signals?.accuracy_drop)}</strong>
            </div>
            <div>
              <span>AUC drop</span>
              <strong>{formatMetric(result?.verdict?.signals?.auc_drop)}</strong>
            </div>
          </div>
        </header>

        <section className="summary-row">
          {summaryTiles.length ? (
            summaryTiles.map((tile) => (
              <MetricTile key={tile.label} label={tile.label} value={tile.value} caption={tile.caption} />
            ))
          ) : (
            <>
              <MetricTile label="Mean Accuracy" value="-" caption="summary appears after a completed run" />
              <MetricTile label="Mean F1" value="-" caption="watch downstream performance over time" />
              <MetricTile label="Mean ROC-AUC" value="-" caption="calibration and ranking quality" />
              <MetricTile label="Train / Test" value="-" caption="dataset split metadata" />
            </>
          )}
        </section>

        {result?.batches?.length ? (
          <>
            <div className="toolbar">
              <nav className="tab-row" aria-label="Observatory views">
                {TABS.map((tab) => (
                  <button
                    key={tab.id}
                    className={tab.id === activeTab ? "tab-button tab-button--active" : "tab-button"}
                    type="button"
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>

              <label className="batch-picker">
                <span>Active batch</span>
                <select
                  value={selectedBatchIndex}
                  onChange={(event) => setSelectedBatchIndex(Number(event.target.value))}
                >
                  {result.batches.map((batch, index) => (
                    <option key={batch.index} value={index}>
                      Batch {batch.index + 1}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {activeTab === "embedding" ? (
              <div className="content-grid content-grid--wide">
                <SectionCard
                  title="Embedding scatter"
                  subtitle="Baseline vs. drifted 2D points for the selected batch."
                >
                  <div className="plot-frame plot-frame--large">
                    <Plot
                      data={[
                        {
                          x: selectedBatch.baseline_emb_2d.map((point) => point[0]),
                          y: selectedBatch.baseline_emb_2d.map((point) => point[1]),
                          mode: "markers",
                          type: "scattergl",
                          name: "Baseline",
                          marker: { color: "#2563eb", size: 7, opacity: 0.45 },
                        },
                        {
                          x: selectedBatch.emb_2d.map((point) => point[0]),
                          y: selectedBatch.emb_2d.map((point) => point[1]),
                          mode: "markers",
                          type: "scattergl",
                          name: "Drifted",
                          marker: { color: "#f97316", size: 7, opacity: 0.62 },
                        },
                      ]}
                      layout={plotLayout("Embedding projection")}
                      config={plotConfig}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                    />
                  </div>
                </SectionCard>

                <SectionCard
                  title={`Batch ${selectedBatch.index + 1} configuration`}
                  subtitle="The currently selected simulated deployment scenario."
                >
                  <dl className="config-grid">
                    <div>
                      <dt>Drift score</dt>
                      <dd>{formatMetric(selectedBatch.drift_score)}</dd>
                    </div>
                    <div>
                      <dt>Covariate strength</dt>
                      <dd>{formatMetric(selectedBatch.config.covariate_strength, 2)}</dd>
                    </div>
                    <div>
                      <dt>Noise std</dt>
                      <dd>{formatMetric(selectedBatch.config.noise_std, 2)}</dd>
                    </div>
                    <div>
                      <dt>Concept flip</dt>
                      <dd>{formatMetric(selectedBatch.config.concept_flip_rate, 2)}</dd>
                    </div>
                    <div>
                      <dt>Prior ratio</dt>
                      <dd>{selectedBatch.config.prior_ratio ?? "disabled"}</dd>
                    </div>
                    <div>
                      <dt>Missing rate</dt>
                      <dd>{formatMetric(selectedBatch.config.missing_rate, 2)}</dd>
                    </div>
                  </dl>
                </SectionCard>
              </div>
            ) : null}

            {activeTab === "clusters" ? (
              <div className="content-grid">
                <SectionCard
                  title="Centroid shifts by batch"
                  subtitle="Each bar group shows how far every cluster center moved."
                >
                  <div className="plot-frame">
                    <Plot
                      data={selectedBatch.centroid_shifts.map((_, clusterIndex) => ({
                        x: timeline.indices,
                        y: result.batches.map((batch) => batch.centroid_shifts[clusterIndex] ?? null),
                        type: "bar",
                        name: `Cluster ${clusterIndex + 1}`,
                      }))}
                      layout={plotLayout("Cluster movement", { barmode: "group" })}
                      config={plotConfig}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                    />
                  </div>
                </SectionCard>

                <SectionCard
                  title="Reassignment rate"
                  subtitle="How often points switch cluster assignments across drift batches."
                >
                  <div className="plot-frame">
                    <Plot
                      data={[
                        {
                          x: timeline.indices,
                          y: timeline.reassignment,
                          type: "scatter",
                          mode: "lines+markers",
                          line: { color: "#f97316", width: 3 },
                          marker: { size: 8 },
                          name: "Reassignment",
                        },
                      ]}
                      layout={plotLayout("Cluster reassignment")}
                      config={plotConfig}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                    />
                  </div>
                </SectionCard>
              </div>
            ) : null}

            {activeTab === "retrieval" ? (
              <div className="content-grid">
                <SectionCard
                  title="Neighbor instability"
                  subtitle="Nearest-neighbor relationships breaking over time."
                >
                  <div className="plot-frame">
                    <Plot
                      data={[
                        {
                          x: timeline.indices,
                          y: timeline.neighborInstability,
                          type: "scatter",
                          mode: "lines+markers",
                          line: { color: "#ef4444", width: 3 },
                          marker: { size: 8 },
                          name: "Neighbor instability",
                        },
                        {
                          x: timeline.indices,
                          y: timeline.drift,
                          type: "scatter",
                          mode: "lines+markers",
                          line: { color: "#2563eb", width: 3, dash: "dot" },
                          marker: { size: 7 },
                          name: "Composite drift score",
                        },
                      ]}
                      layout={plotLayout("Retrieval stability")}
                      config={plotConfig}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                    />
                  </div>
                </SectionCard>

                <SectionCard
                  title="Selected batch retrieval signals"
                  subtitle="Fine-grained geometry signals for the active batch."
                >
                  <div className="signal-list">
                    <div>
                      <span>Neighbor instability</span>
                      <strong>{formatMetric(selectedBatch.neighbor_instability)}</strong>
                    </div>
                    <div>
                      <span>Mean cosine shift</span>
                      <strong>{formatMetric(selectedBatch.mean_cosine_shift)}</strong>
                    </div>
                    <div>
                      <span>Mean euclidean shift</span>
                      <strong>{formatMetric(selectedBatch.mean_euclidean_shift)}</strong>
                    </div>
                    <div>
                      <span>Mean centroid shift</span>
                      <strong>{formatMetric(selectedBatch.mean_centroid_shift)}</strong>
                    </div>
                  </div>
                </SectionCard>
              </div>
            ) : null}

            {activeTab === "performance" ? (
              <div className="content-grid">
                <SectionCard
                  title="Performance over time"
                  subtitle="Accuracy, F1, and ROC-AUC across drift batches."
                >
                  <div className="plot-frame">
                    <Plot
                      data={[
                        series("Accuracy", timeline.indices, timeline.accuracy, "#0f766e"),
                        series("F1", timeline.indices, timeline.f1, "#1d4ed8"),
                        series("ROC-AUC", timeline.indices, timeline.rocAuc, "#d97706"),
                      ]}
                      layout={plotLayout("Model performance")}
                      config={plotConfig}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                    />
                  </div>
                </SectionCard>

                <SectionCard
                  title={`Calibration for batch ${selectedBatch.index + 1}`}
                  subtitle="Well-calibrated predictions should sit close to the diagonal."
                >
                  <div className="plot-frame">
                    <Plot
                      data={[
                        {
                          x: [0, 1],
                          y: [0, 1],
                          type: "scatter",
                          mode: "lines",
                          name: "Perfect calibration",
                          line: { color: "#94a3b8", dash: "dash" },
                        },
                        {
                          x: selectedBatch.calibration.mean_predicted_value,
                          y: selectedBatch.calibration.fraction_of_positives,
                          type: "scatter",
                          mode: "lines+markers",
                          name: "Model",
                          line: { color: "#7c3aed", width: 3 },
                          marker: { size: 8 },
                        },
                      ]}
                      layout={plotLayout("Calibration curve", {
                        xaxis: { title: "Mean predicted value", range: [0, 1] },
                        yaxis: { title: "Fraction of positives", range: [0, 1] },
                      })}
                      config={plotConfig}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                    />
                  </div>
                </SectionCard>
              </div>
            ) : null}
          </>
        ) : (
          <section className="empty-state">
            <p className="eyebrow">Awaiting results</p>
            <h3>Run the full pipeline to populate the observatory.</h3>
            <p>
              The React dashboard is wired for the backend in `FRONTEND.md`: upload a CSV,
              choose the label, adjust drift settings, and launch a run to see the embedding,
              cluster, retrieval, and performance panels.
            </p>
          </section>
        )}
      </section>
    </main>
  )
}

function series(name, x, y, color) {
  return {
    x,
    y,
    type: "scatter",
    mode: "lines+markers",
    name,
    line: { color, width: 3 },
    marker: { size: 8 },
  }
}

const plotConfig = {
  displayModeBar: false,
  responsive: true,
}

function plotLayout(title, overrides = {}) {
  const base = {
    title: {
      text: title,
      x: 0,
      font: { size: 16, color: "#f8fafc" },
    },
    autosize: true,
    margin: { l: 52, r: 24, t: 46, b: 48 },
    paper_bgcolor: "#0f172a",
    plot_bgcolor: "#0f172a",
    font: { color: "#cbd5e1" },
    xaxis: {
      title: "Batch",
      gridcolor: "rgba(148, 163, 184, 0.16)",
      zerolinecolor: "rgba(148, 163, 184, 0.2)",
    },
    yaxis: {
      title: "Value",
      gridcolor: "rgba(148, 163, 184, 0.16)",
      zerolinecolor: "rgba(148, 163, 184, 0.2)",
    },
    legend: {
      orientation: "h",
      y: 1.14,
      x: 0,
    },
  }

  return {
    ...base,
    ...overrides,
    xaxis: {
      ...base.xaxis,
      ...(overrides.xaxis || {}),
    },
    yaxis: {
      ...base.yaxis,
      ...(overrides.yaxis || {}),
    },
  }
}
