import React, { useState } from "react";
import Plot from "react-plotly.js";
import type Plotly from "plotly.js";
import { api, CompatResult, MetaValidationResult, TrainParams } from "../../api";
import ServerOrUpload from "../ServerOrUpload";

export interface ModelEntry {
  name: string;
  source: "uploaded" | "trained";
  compat?: CompatResult;
  trainParams?: TrainParams;
  status: "ready" | "training" | "error";
  testAccuracy?: number;
  dpsgd?: boolean;
  targetEpsilon?: number;
}

interface Props {
  jobId: string;
  onDone: (models: ModelEntry[]) => void;
}

export default function Step4Models({ jobId, onDone }: Props) {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [showUpload, setShowUpload] = useState(false);
  const [showTrain, setShowTrain] = useState(false);

  const addModel = (m: ModelEntry) =>
    setModels((prev) => [...prev.filter((x) => x.name !== m.name), m]);

  const updateModel = (name: string, patch: Partial<ModelEntry>) =>
    setModels((prev) => prev.map((m) => m.name === name ? { ...m, ...patch } : m));

  const canProceed = models.length > 0 && models.every((m) => m.status === "ready");

  return (
    <div className="flex flex-col gap-8">
      <div className="space-y-2">
        <h2 className="text-4xl font-black tracking-tight">Models</h2>
        <p className="text-slate-600 dark:text-slate-400 text-lg max-w-2xl">
          Upload an existing trained model, train new ones, or both. You can add multiple models
          to compare them side-by-side in the results.
        </p>
      </div>

      {/* Sub-flow A: upload existing */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
        <button
          onClick={() => setShowUpload(!showUpload)}
          className="w-full flex items-center gap-3 px-6 py-4 bg-slate-50/50 dark:bg-slate-900/50 hover:bg-primary/5 transition-colors text-left"
        >
          <span className="material-symbols-outlined text-primary">upload</span>
          <span className="font-bold">I have a trained model to upload</span>
          <span className="ml-auto material-symbols-outlined text-slate-400">
            {showUpload ? "expand_less" : "expand_more"}
          </span>
        </button>
        {showUpload && (
          <div className="p-6">
            <UploadModelForm jobId={jobId} onAdded={addModel} onUpdated={updateModel} existingNames={models.map((m) => m.name)} />
          </div>
        )}
      </div>

      {/* Sub-flow B: train new */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
        <button
          onClick={() => setShowTrain(!showTrain)}
          className="w-full flex items-center gap-3 px-6 py-4 bg-slate-50/50 dark:bg-slate-900/50 hover:bg-primary/5 transition-colors text-left"
        >
          <span className="material-symbols-outlined text-primary">model_training</span>
          <span className="font-bold">Train a new model</span>
          <span className="ml-auto material-symbols-outlined text-slate-400">
            {showTrain ? "expand_less" : "expand_more"}
          </span>
        </button>
        {showTrain && (
          <div className="p-6">
            <TrainModelForm jobId={jobId} onAdded={addModel} existingCount={models.filter(m => m.source === "trained").length} />
          </div>
        )}
      </div>

      {/* Model list */}
      {models.length > 0 && (
        <div className="flex flex-col gap-3">
          <h3 className="font-bold text-sm uppercase tracking-wider text-slate-500">Models ready for audit</h3>
          {models.map((m) => <ModelCard key={m.name} model={m} onRemove={() => setModels(prev => prev.filter(x => x.name !== m.name))} />)}
        </div>
      )}

      <div className="flex justify-end">
        <button
          onClick={() => onDone(models)}
          disabled={!canProceed}
          className="px-8 py-2.5 rounded-lg bg-primary text-white font-bold hover:bg-primary/90 transition-colors flex items-center gap-2 shadow-lg shadow-primary/20 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Continue to Attack Config
          <span className="material-symbols-outlined text-base">arrow_forward</span>
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload sub-form — two independent validation buttons, supports multiple models
// ---------------------------------------------------------------------------
function UploadModelForm({ jobId, onAdded, onUpdated, existingNames }: {
  jobId: string;
  onAdded: (m: ModelEntry) => void;
  onUpdated: (name: string, patch: Partial<ModelEntry>) => void;
  existingNames: string[];
}) {
  const nextName = () => {
    const base = "uploaded_model";
    const taken = new Set(existingNames);
    if (!taken.has(base)) return base;
    let n = 2;
    while (taken.has(`${base}_${n}`)) n++;
    return `${base}_${n}`;
  };

  const [name, setName] = useState(nextName);
  // Track upload completion via onDone callback (more reliable than async state in onFile)
  const [weightsUploaded, setWeightsUploaded] = useState(false);
  const [metaUploaded, setMetaUploaded] = useState(false);
  // Model validation (weights check + inference)
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<CompatResult | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);
  const [addedToList, setAddedToList] = useState(false);
  // Metadata validation (field presence check)
  const [metaValidating, setMetaValidating] = useState(false);
  const [metaValidation, setMetaValidation] = useState<MetaValidationResult | null>(null);

  const reset = () => {
    setWeightsUploaded(false);
    setMetaUploaded(false);
    setChecking(false);
    setResult(null);
    setCheckError(null);
    setAddedToList(false);
    setMetaValidating(false);
    setMetaValidation(null);
    setMetaError(null);
    setName(nextName());
  };

  const doValidateModel = async () => {
    setChecking(true);
    setCheckError(null);
    setResult(null);
    // Register immediately in the audit list so it shows up right away
    if (!addedToList) {
      onAdded({ name, source: "uploaded", status: "error" });
    }
    try {
      const r = await api.checkCompat(jobId, name);
      setResult(r);
      if (r.ok) {
        onUpdated(name, { compat: r, status: "ready" });
        setAddedToList(true);
      } else {
        onUpdated(name, { status: "error" });
      }
    } catch (e: unknown) {
      setCheckError(e instanceof Error ? e.message : String(e));
      onUpdated(name, { status: "error" });
    } finally {
      setChecking(false);
    }
  };

  const [metaError, setMetaError] = useState<string | null>(null);

  const doValidateMeta = async () => {
    setMetaValidating(true);
    setMetaError(null);
    try {
      const v = await api.validateModelMetadata(jobId, name);
      setMetaValidation(v);
    } catch (e: unknown) {
      setMetaError(e instanceof Error ? e.message : String(e));
    } finally { setMetaValidating(false); }
  };

  // Lock the name once any file is uploaded so upload and validate always match
  const nameLocked = weightsUploaded || metaUploaded;

  return (
    <div className="flex flex-col gap-4">
      {/* Model name — locked once a file is uploaded */}
      <div>
        <label className="text-xs font-semibold text-slate-500 mb-1 block">Model name</label>
        <input
          value={name}
          disabled={nameLocked}
          onChange={(e) => { setName(e.target.value); setResult(null); setCheckError(null); }}
          className="w-full rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm px-3 py-2 disabled:opacity-60 disabled:cursor-not-allowed"
        />
      </div>

      {/* ── Model weights ─────────────────────────────────────── */}
      {!result?.ok && (
        <div className="rounded-lg border border-slate-200 dark:border-slate-700 p-4 space-y-3">
          <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Model weights</p>
          <ServerOrUpload
            label="Weights file (.pkl / .pt / .pth)"
            hint="Saved with torch.save(model.state_dict(), path)"
            icon="save"
            accept=".pkl,.pt,.pth"
            onFile={async (f) => { setResult(null); setCheckError(null); await api.uploadWeights(jobId, name, f); }}
            onPath={async (p) => { setResult(null); setCheckError(null); await api.setWeightsPath(jobId, name, p); }}
            onDone={() => setWeightsUploaded(true)}
          />
          {weightsUploaded && (
            <button
              onClick={doValidateModel}
              disabled={checking}
              className="flex items-center gap-2 px-5 py-2 rounded-lg bg-primary text-white font-bold text-sm hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              {checking
                ? <><span className="material-symbols-outlined text-base animate-spin">sync</span> Validating…</>
                : result
                  ? <><span className="material-symbols-outlined text-base">refresh</span> Retry Validation</>
                  : <><span className="material-symbols-outlined text-base">speed</span> Validate Model</>}
            </button>
          )}
          {result && <CompatCard result={result} />}
          {checkError && <p className="text-sm text-red-500 font-mono text-xs whitespace-pre-wrap">{checkError}</p>}
        </div>
      )}

      {/* Success banner — shown after model validates */}
      {result?.ok && (
        <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-3 flex items-center gap-3">
          <span className="material-symbols-outlined text-green-500">check_circle</span>
          <span className="text-sm font-semibold text-green-700 dark:text-green-400 flex-1">
            <strong>{name}</strong> validated and added to the audit list.
          </span>
        </div>
      )}

      {/* ── Model metadata — always visible once uploaded, survives model validation ── */}
      <div className="rounded-lg border border-slate-200 dark:border-slate-700 p-4 space-y-3">
        <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Model metadata</p>
        <ServerOrUpload
          label="Metadata file (.pkl)"
          hint="Contains train/test indices, optimizer, criterion, and training results"
          icon="description"
          accept=".pkl"
          onFile={async (f) => { setMetaValidation(null); await api.uploadModelMetadata(jobId, name, f); }}
          onPath={async (p) => { setMetaValidation(null); await api.setMetadataPath(jobId, name, p); }}
          onDone={() => setMetaUploaded(true)}
        />
        {metaUploaded && (
          <button
            onClick={doValidateMeta}
            disabled={metaValidating}
            className="flex items-center gap-2 px-5 py-2 rounded-lg border border-primary text-primary font-bold text-sm hover:bg-primary/5 transition-colors disabled:opacity-50"
          >
            {metaValidating
              ? <><span className="material-symbols-outlined text-base animate-spin">sync</span> Validating…</>
              : metaValidation
                ? <><span className="material-symbols-outlined text-base">refresh</span> Re-validate Metadata</>
                : <><span className="material-symbols-outlined text-base">fact_check</span> Validate Metadata</>}
          </button>
        )}
        {metaValidation && <MetaValidationCard result={metaValidation} />}
        {metaError && <p className="text-xs text-red-500 font-mono whitespace-pre-wrap">{metaError}</p>}
      </div>

      {/* Add another model — shown after success */}
      {result?.ok && (
        <button onClick={reset} className="flex items-center gap-1.5 text-primary text-sm font-bold hover:underline self-start">
          <span className="material-symbols-outlined text-base">add_circle</span>
          Add another model
        </button>
      )}
    </div>
  );
}

function MetaValidationCard({ result }: { result: MetaValidationResult }) {
  if (result.ok) {
    return (
      <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-3 flex items-start gap-2">
        <span className="material-symbols-outlined text-green-500 text-base mt-0.5">check_circle</span>
        <div>
          <p className="text-sm font-semibold text-green-700 dark:text-green-400">Metadata valid — all required fields present</p>
          <p className="text-xs text-slate-500 mt-0.5">{result.present_fields.join(", ")}</p>
        </div>
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 flex items-start gap-2">
      <span className="material-symbols-outlined text-amber-500 text-base mt-0.5">warning</span>
      <div>
        <p className="text-sm font-semibold text-amber-700 dark:text-amber-400">Metadata incomplete</p>
        {result.missing_fields.length > 0 && (
          <p className="text-xs text-slate-500 mt-0.5">Missing: <span className="font-mono">{result.missing_fields.join(", ")}</span></p>
        )}
        {result.error && <pre className="text-xs text-red-500 mt-1 whitespace-pre-wrap">{result.error.slice(0, 300)}</pre>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Train sub-form
// ---------------------------------------------------------------------------
interface TrainMetrics {
  model: string;
  loss_history: number[];
  accuracy_history: number[];
  val_loss_history?: number[];
  val_acc_history?: number[];
  test_loss_final?: number | null;
  test_acc_final?: number | null;
}

function TrainModelForm({ jobId, onAdded, existingCount }: {
  jobId: string; onAdded: (m: ModelEntry) => void; existingCount: number;
}) {
  const [cards, setCards] = useState([defaultParams(existingCount)]);
  const [logs, setLogs] = useState<string[]>([]);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState<{ epoch: number; total: number; model: string } | null>(null);
  const [metrics, setMetrics] = useState<TrainMetrics[]>([]);
  const logEndRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const addCard = () => setCards((prev) => [...prev, defaultParams(prev.length + existingCount + metrics.length)]);
  const removeCard = (i: number) => setCards((prev) => prev.filter((_, j) => j !== i));
  const update = (i: number, patch: Partial<TrainParams>) =>
    setCards((prev) => prev.map((c, j) => j === i ? { ...c, ...patch } : c));

  const trainAll = async () => {
    setTraining(true);
    setLogs([]);
    setProgress(null);

    // Only train the cards that are currently shown (not already-trained models)
    const toTrain = [...cards];

    const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/jobs/${jobId}/logs`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (e) => {
      const msg: string = e.data;
      if (msg.startsWith("__STATUS__")) return;
      if (msg.startsWith("__PROGRESS__")) {
        const rest = msg.slice("__PROGRESS__".length);
        const [modelName, epochStr, totalStr] = rest.split("|");
        setProgress({ model: modelName, epoch: Number(epochStr), total: Number(totalStr) });
        return;
      }
      if (msg.startsWith("__METRICS__")) {
        try {
          const m: TrainMetrics = JSON.parse(msg.slice("__METRICS__".length));
          setMetrics((prev) => [...prev.filter((x) => x.model !== m.model), m]);
        } catch { /* ignore */ }
        return;
      }
      if (msg === "__TRAIN_DONE__") {
        ws.close();
        setTraining(false);
        setProgress(null);
        for (const params of toTrain) {
          onAdded({ name: params.name, source: "trained", status: "ready",
            dpsgd: params.dpsgd, targetEpsilon: params.target_epsilon, trainParams: params });
        }
        // Reset form to a fresh card (prevents re-training already-trained models)
        setCards([defaultParams(existingCount + toTrain.length)]);
        return;
      }
      setLogs((prev) => [...prev, msg]);
    };

    ws.onclose = () => { setTraining(false); setProgress(null); };

    for (const params of toTrain) {
      // Mark only this batch as "training" — don't touch already-ready models
      onAdded({ name: params.name, source: "trained", status: "training",
        dpsgd: params.dpsgd, targetEpsilon: params.target_epsilon, trainParams: params });
      await api.trainModel(jobId, params);
    }
  };

  const pct = progress ? Math.round((progress.epoch / progress.total) * 100) : 0;

  return (
    <div className="flex flex-col gap-6">
      {cards.map((p, i) => (
        <TrainCard key={i} params={p} onChange={(patch) => update(i, patch)}
          onRemove={cards.length > 1 ? () => removeCard(i) : undefined} />
      ))}

      <div className="flex gap-3 items-center">
        <button onClick={addCard} disabled={training}
          className="flex items-center gap-2 text-primary text-sm font-bold hover:underline disabled:opacity-40"
        >
          <span className="material-symbols-outlined text-base">add_circle</span>
          Add another model
        </button>
        <button onClick={trainAll} disabled={training}
          className="ml-auto px-6 py-2 rounded-lg bg-primary text-white font-bold text-sm hover:bg-primary/90 transition-colors shadow-lg shadow-primary/20 disabled:opacity-50 flex items-center gap-2"
        >
          {training && <span className="material-symbols-outlined text-base animate-spin">sync</span>}
          {training ? "Training…" : "Start Training"}
        </button>
      </div>

      {/* Epoch progress bar */}
      {training && progress && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span className="font-semibold">{progress.model} — Epoch {progress.epoch} / {progress.total}</span>
            <span className="font-mono">{pct}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
            <div
              className="h-full rounded-full bg-primary transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {/* Live log stream */}
      {(logs.length > 0 || training) && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
          <div className="px-4 py-2 bg-slate-100 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-xs font-mono text-slate-500 flex items-center gap-2">
            <span className={`size-2 rounded-full ${training ? "bg-green-400 animate-pulse" : "bg-slate-400"}`} />
            Training log
          </div>
          <div className="bg-slate-950 p-4 h-48 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5">
            {logs.length === 0 && <span className="text-slate-600 animate-pulse">Waiting for output…</span>}
            {logs.map((line, i) => <div key={i}>{line}</div>)}
            <div ref={logEndRef} />
          </div>
        </div>
      )}

      {/* Per-model metrics charts */}
      {metrics.map((m) => (
        <TrainMetricsChart key={m.model} metrics={m} />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Training metrics chart
// ---------------------------------------------------------------------------
function TrainMetricsChart({ metrics }: { metrics: TrainMetrics }) {
  const epochs = metrics.loss_history.map((_, i) => i + 1);
  const darkMode = document.documentElement.classList.contains("dark");
  const gridColor = darkMode ? "#334155" : "#e2e8f0";
  const fontColor = darkMode ? "#94a3b8" : "#64748b";

  const layout: Partial<Plotly.Layout> = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 30, r: 20, b: 40, l: 45 },
    height: 200,
    legend: { orientation: "h", y: -0.3, font: { size: 11, color: fontColor } },
    xaxis: { title: { text: "Epoch" }, gridcolor: gridColor, color: fontColor, tickfont: { size: 10 } },
    yaxis: { gridcolor: gridColor, color: fontColor, tickfont: { size: 10 } },
    font: { family: "Inter, sans-serif" },
  };

  // Build loss traces
  const lossData: Plotly.Data[] = [
    { x: epochs, y: metrics.loss_history, type: "scatter", mode: "lines+markers",
      name: "Train", line: { color: "#193ce6", width: 2 }, marker: { size: 4 } } as Plotly.Data,
  ];
  if (metrics.val_loss_history?.length) {
    const valEpochs = metrics.val_loss_history.map((_, i) => i + 1);
    lossData.push({ x: valEpochs, y: metrics.val_loss_history, type: "scatter", mode: "lines+markers",
      name: "Val", line: { color: "#f28e2b", width: 2, dash: "dash" }, marker: { size: 4 } } as Plotly.Data);
  } else if (metrics.test_loss_final != null) {
    lossData.push({ x: [epochs.length], y: [metrics.test_loss_final], type: "scatter", mode: "markers",
      name: "Test (final)", marker: { color: "#f28e2b", size: 10, symbol: "star" } } as Plotly.Data);
  }

  // Build accuracy traces
  const accData: Plotly.Data[] = [
    { x: epochs, y: metrics.accuracy_history.map((v) => v * 100), type: "scatter", mode: "lines+markers",
      name: "Train", line: { color: "#22c55e", width: 2 }, marker: { size: 4 } } as Plotly.Data,
  ];
  if (metrics.val_acc_history?.length) {
    const valEpochs = metrics.val_acc_history.map((_, i) => i + 1);
    accData.push({ x: valEpochs, y: metrics.val_acc_history.map((v) => v * 100), type: "scatter", mode: "lines+markers",
      name: "Val", line: { color: "#e15759", width: 2, dash: "dash" }, marker: { size: 4 } } as Plotly.Data);
  } else if (metrics.test_acc_final != null) {
    accData.push({ x: [epochs.length], y: [metrics.test_acc_final * 100], type: "scatter", mode: "markers",
      name: "Test (final)", marker: { color: "#e15759", size: 10, symbol: "star" } } as Plotly.Data);
  }

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
      <div className="px-4 py-2 bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-xs font-semibold text-slate-500 flex items-center gap-2">
        <span className="material-symbols-outlined text-sm text-green-500">show_chart</span>
        {metrics.model} — Training curves
      </div>
      <div className="grid grid-cols-2 divide-x divide-slate-200 dark:divide-slate-800 bg-white dark:bg-slate-950">
        <Plot
          data={lossData}
          layout={{ ...layout, title: { text: "Loss", font: { size: 12, color: fontColor } } }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
        <Plot
          data={accData}
          layout={{ ...layout, title: { text: "Accuracy (%)", font: { size: 12, color: fontColor } },
            yaxis: { ...layout.yaxis, range: [0, 100] } }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
      </div>
    </div>
  );
}

function TrainCard({ params, onChange, onRemove }: {
  params: TrainParams; onChange: (p: Partial<TrainParams>) => void; onRemove?: () => void;
}) {
  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <input
          value={params.name}
          onChange={(e) => onChange({ name: e.target.value })}
          className="font-bold text-sm border-0 border-b border-slate-300 dark:border-slate-600 bg-transparent focus:outline-none focus:border-primary px-0 py-1"
          placeholder="Model name"
        />
        {onRemove && (
          <button onClick={onRemove} className="text-slate-400 hover:text-red-500 transition-colors">
            <span className="material-symbols-outlined text-base">close</span>
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <NumberField label="Epochs" value={params.epochs} min={1} max={500} onChange={(v) => onChange({ epochs: v })} />
        <NumberField label="Learning Rate" value={params.learning_rate} min={0.0001} max={1} step={0.0001} onChange={(v) => onChange({ learning_rate: v })} />
        <NumberField label="Batch Size" value={params.batch_size} min={8} max={512} onChange={(v) => onChange({ batch_size: v })} />
        <div>
          <label className="text-xs font-semibold text-slate-500 mb-1 block">Optimizer</label>
          <select
            value={params.optimizer}
            onChange={(e) => onChange({ optimizer: e.target.value })}
            className="w-full rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm px-3 py-2"
          >
            <option value="adam">Adam</option>
            <option value="sgd">SGD</option>
          </select>
        </div>
      </div>

      {/* Data split */}
      <div className="grid grid-cols-2 gap-3">
        <NumberField label="Train fraction (f_train)" value={params.f_train ?? 0.5} min={0.1} max={0.9} step={0.05}
          onChange={(v) => onChange({ f_train: v })} />
        <NumberField label="Test fraction (f_test)" value={params.f_test ?? 0.5} min={0.1} max={0.9} step={0.05}
          onChange={(v) => onChange({ f_test: v })} />
      </div>

      {/* DP-SGD toggle */}
      <label className="flex items-center gap-3 cursor-pointer">
        <div
          onClick={() => onChange({ dpsgd: !params.dpsgd })}
          className={`relative w-10 h-5 rounded-full transition-colors ${params.dpsgd ? "bg-primary" : "bg-slate-300 dark:bg-slate-600"}`}
        >
          <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${params.dpsgd ? "translate-x-5" : ""}`} />
        </div>
        <span className="text-sm font-semibold">Train with DP-SGD</span>
        {params.dpsgd && <span className="text-xs text-slate-500">(differential privacy)</span>}
      </label>

      {params.dpsgd && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pl-4 border-l-2 border-primary/30">
          <NumberField label="Target ε" value={params.target_epsilon ?? 10} min={0.1} max={100} step={0.1} onChange={(v) => onChange({ target_epsilon: v })} />
          <NumberField label="Target δ" value={params.target_delta ?? 1e-5} min={1e-7} max={0.1} step={1e-6} onChange={(v) => onChange({ target_delta: v })} />
          <NumberField label="Max Grad Norm" value={params.max_grad_norm ?? 1.0} min={0.01} max={10} step={0.01} onChange={(v) => onChange({ max_grad_norm: v })} />
          <NumberField label="Virtual Batch Size" value={params.virtual_batch_size ?? 16} min={1} max={512} step={1} onChange={(v) => onChange({ virtual_batch_size: v })} />
        </div>
      )}
    </div>
  );
}

function NumberField({ label, value, min, max, step = 1, onChange }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="text-xs font-semibold text-slate-500 mb-1 block">{label}</label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm px-3 py-2"
      />
    </div>
  );
}

function CompatCard({ result }: { result: CompatResult }) {
  if (result.ok) {
    return (
      <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-4 space-y-3">
        <div className="flex items-center gap-2 text-green-600 dark:text-green-400 font-bold">
          <span className="material-symbols-outlined">check_circle</span>
          Model compatible
        </div>

        {/* Shape + param info */}
        <div className="grid grid-cols-3 gap-3 text-sm">
          <div><span className="text-slate-500">Input shape: </span><span className="font-mono font-semibold">[{result.input_shape?.join(", ")}]</span></div>
          <div><span className="text-slate-500">Output shape: </span><span className="font-mono font-semibold">[{result.output_shape?.join(", ")}]</span></div>
          <div><span className="text-slate-500">Parameters: </span><span className="font-mono font-semibold">{result.param_count?.toLocaleString()}</span></div>
        </div>

        {/* Inference test results */}
        {result.sample_outputs && result.sample_outputs.length > 0 && (
          <div className="rounded-md border border-green-500/20 bg-white/60 dark:bg-slate-900/60 p-3 space-y-2">
            <p className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
              <span className="material-symbols-outlined text-sm text-primary">psychology</span>
              Inference test — real data samples
            </p>
            {result.sample_outputs.map((s) => {
              const match = s.true_label !== null && s.top1_class === s.true_label;
              const hasLabel = s.true_label !== null;
              return (
                <div key={s.sample} className="flex items-center gap-2 text-sm">
                  <span className="text-slate-400 text-xs w-16 shrink-0">Sample {s.sample + 1}</span>
                  <span className="text-slate-600 dark:text-slate-300">
                    predicted class <span className="font-bold font-mono">{s.top1_class}</span>
                    {" "}
                    <span className="text-slate-400">({(s.confidence * 100).toFixed(1)}% confidence)</span>
                  </span>
                  {hasLabel && (
                    <>
                      <span className="text-slate-400 text-xs">· true: {s.true_label}</span>
                      <span className={`text-xs font-bold ${match ? "text-green-500" : "text-red-400"}`}>
                        {match ? "✓" : "✗"}
                      </span>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-4">
      <div className="flex items-center gap-2 text-red-600 dark:text-red-400 font-bold mb-2">
        <span className="material-symbols-outlined">error</span>
        Compatibility check failed
      </div>
      <pre className="text-xs text-red-500 whitespace-pre-wrap font-mono">{result.error}</pre>
    </div>
  );
}

function ModelCard({ model, onRemove }: { model: ModelEntry; onRemove: () => void }) {
  return (
    <div className="flex items-center gap-4 p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50">
      <div className={`size-9 rounded-lg flex items-center justify-center
        ${model.status === "ready" ? "bg-green-500/10 text-green-500" : model.status === "training" ? "bg-primary/10 text-primary" : "bg-red-500/10 text-red-500"}`}>
        <span className="material-symbols-outlined text-base">
          {model.status === "ready" ? "check_circle" : model.status === "training" ? "sync" : "error"}
        </span>
      </div>
      <div className="flex-1">
        <p className="font-bold text-sm">{model.name}</p>
        <p className="text-xs text-slate-500">{model.source === "uploaded" ? "Uploaded" : model.status === "ready" ? "Trained" : "Training…"}{model.dpsgd ? " · DP-SGD (ε=" + model.targetEpsilon + ")" : ""}</p>
      </div>
      <button onClick={onRemove} className="text-slate-400 hover:text-red-500 transition-colors">
        <span className="material-symbols-outlined text-base">close</span>
      </button>
    </div>
  );
}

function defaultParams(n: number): TrainParams {
  return {
    name: `model_${n + 1}`,
    epochs: 50,
    learning_rate: 0.001,
    batch_size: 128,
    optimizer: "adam",
    f_train: 0.5,
    f_test: 0.5,
    dpsgd: false,
  };
}
