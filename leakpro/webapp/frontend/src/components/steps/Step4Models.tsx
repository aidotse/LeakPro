import React, { useState } from "react";
import Plot from "react-plotly.js";
import type Plotly from "plotly.js";
import { api, CompatResult, TrainParams } from "../../api";
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
            <UploadModelForm jobId={jobId} onAdded={addModel} existingNames={models.map((m) => m.name)} />
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
// Upload sub-form
// ---------------------------------------------------------------------------
function UploadModelForm({ jobId, onAdded, existingNames }: {
  jobId: string; onAdded: (m: ModelEntry) => void; existingNames: string[];
}) {
  const [name, setName] = useState("uploaded_model");
  const [weightsFile, setWeightsFile] = useState<File | null>(null);
  const [metaFile, setMetaFile] = useState<File | null>(null);
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<CompatResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const bothUploaded = !!(weightsFile && metaFile);

  const doCheck = async () => {
    setChecking(true);
    setError(null);
    try {
      const r = await api.checkCompat(jobId, name);
      setResult(r);
      if (r.ok) {
        onAdded({ name, source: "uploaded", compat: r, status: "ready" });
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div>
        <label className="text-xs font-semibold text-slate-500 mb-1 block">Model name</label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm px-3 py-2"
        />
      </div>

      <ServerOrUpload
        label="Model weights (.pkl / .pt / .pth)"
        hint="State dict saved with torch.save(model.state_dict(), f)"
        icon="save"
        accept=".pkl,.pt,.pth"
        onFile={async (f) => { setWeightsFile(f); setResult(null); await api.uploadWeights(jobId, name, f); }}
        onPath={async (p) => { setWeightsFile(new File([], p.split("/").pop() ?? "model.pkl")); setResult(null); await api.setWeightsPath(jobId, name, p); }}
      />

      <ServerOrUpload
        label="Model metadata (.pkl)"
        hint="Output of LeakPro.make_mia_metadata() — required for audit"
        icon="description"
        accept=".pkl"
        onFile={async (f) => { setMetaFile(f); setResult(null); await api.uploadModelMetadata(jobId, name, f); }}
        onPath={async (_p) => { /* server-path for metadata not yet supported */ }}
      />

      {bothUploaded && !result && (
        <button
          onClick={doCheck}
          disabled={checking}
          className="self-start px-6 py-2 rounded-lg bg-primary text-white font-bold text-sm hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {checking ? "Checking…" : "Check Compatibility"}
        </button>
      )}

      {!bothUploaded && (weightsFile || metaFile) && (
        <p className="text-xs text-slate-400">
          {!weightsFile ? "Upload model weights to continue." : "Upload model metadata to continue."}
        </p>
      )}

      {result && <CompatCard result={result} />}
      {error && <p className="text-sm text-red-500">{error}</p>}
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
    height: 180,
    legend: { orientation: "h", y: -0.25, font: { size: 11, color: fontColor } },
    xaxis: { title: "Epoch", gridcolor: gridColor, color: fontColor, tickfont: { size: 10 } },
    yaxis: { gridcolor: gridColor, color: fontColor, tickfont: { size: 10 } },
    font: { family: "Inter, sans-serif" },
  };

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
      <div className="px-4 py-2 bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-xs font-semibold text-slate-500 flex items-center gap-2">
        <span className="material-symbols-outlined text-sm text-green-500">show_chart</span>
        {metrics.model} — Training curves
      </div>
      <div className="grid grid-cols-2 divide-x divide-slate-200 dark:divide-slate-800 bg-white dark:bg-slate-950">
        <Plot
          data={[{ x: epochs, y: metrics.loss_history, type: "scatter", mode: "lines+markers",
            name: "Train loss", line: { color: "#193ce6", width: 2 }, marker: { size: 4 } }]}
          layout={{ ...layout, title: { text: "Loss", font: { size: 12, color: fontColor } } }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
        <Plot
          data={[{ x: epochs, y: metrics.accuracy_history.map((v) => v * 100), type: "scatter", mode: "lines+markers",
            name: "Train accuracy", line: { color: "#22c55e", width: 2 }, marker: { size: 4 } }]}
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
        <div className="grid grid-cols-3 gap-3 pl-4 border-l-2 border-primary/30">
          <NumberField label="Target ε" value={params.target_epsilon ?? 10} min={0.1} max={100} step={0.1} onChange={(v) => onChange({ target_epsilon: v })} />
          <NumberField label="Target δ" value={params.target_delta ?? 1e-5} min={1e-7} max={0.1} step={1e-6} onChange={(v) => onChange({ target_delta: v })} />
          <NumberField label="Max Grad Norm" value={params.max_grad_norm ?? 1.0} min={0.01} max={10} step={0.01} onChange={(v) => onChange({ max_grad_norm: v })} />
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
      <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-4 space-y-2">
        <div className="flex items-center gap-2 text-green-600 dark:text-green-400 font-bold">
          <span className="material-symbols-outlined">check_circle</span>
          Compatibility check passed
        </div>
        <div className="grid grid-cols-3 gap-3 text-sm">
          <div><span className="text-slate-500">Input: </span><span className="font-mono">[{result.input_shape?.join(", ")}]</span></div>
          <div><span className="text-slate-500">Output: </span><span className="font-mono">[{result.output_shape?.join(", ")}]</span></div>
          <div><span className="text-slate-500">Params: </span><span className="font-mono">{result.param_count?.toLocaleString()}</span></div>
        </div>
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
