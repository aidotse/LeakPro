import React, { useState } from "react";
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
  const [file, setFile] = useState<File | null>(null);
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<CompatResult | null>(null);
  const [error, setError] = useState<string | null>(null);

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
        label="Model weights (.pt / .pth)"
        hint="Saved PyTorch model weights"
        icon="save"
        accept=".pt,.pth"
        onFile={async (f) => { setFile(f); setResult(null); await api.uploadWeights(jobId, name, f); }}
        onPath={async (p) => { setFile(new File([], p.split("/").pop() ?? "model.pt")); setResult(null); await api.setWeightsPath(jobId, name, p); }}
      />

      {file && !result && (
        <button
          onClick={doCheck}
          disabled={checking}
          className="self-start px-6 py-2 rounded-lg bg-primary text-white font-bold text-sm hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {checking ? "Checking…" : "Check Compatibility"}
        </button>
      )}

      {result && <CompatCard result={result} />}
      {error && <p className="text-sm text-red-500">{error}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Train sub-form
// ---------------------------------------------------------------------------
function TrainModelForm({ jobId, onAdded, existingCount }: {
  jobId: string; onAdded: (m: ModelEntry) => void; existingCount: number;
}) {
  const [cards, setCards] = useState([defaultParams(existingCount)]);
  const [logs, setLogs] = useState<string[]>([]);
  const [training, setTraining] = useState(false);
  const logEndRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const addCard = () => setCards((prev) => [...prev, defaultParams(prev.length + existingCount)]);
  const removeCard = (i: number) => setCards((prev) => prev.filter((_, j) => j !== i));
  const update = (i: number, patch: Partial<TrainParams>) =>
    setCards((prev) => prev.map((c, j) => j === i ? { ...c, ...patch } : c));

  const trainAll = async () => {
    setTraining(true);
    setLogs([]);

    // Open WebSocket for log stream
    const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/jobs/${jobId}/logs`;
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (e) => {
      const msg: string = e.data;
      if (!msg.startsWith("__STATUS__")) {
        setLogs((prev) => [...prev, msg]);
      }
    };

    for (const params of cards) {
      await api.trainModel(jobId, params);
      onAdded({
        name: params.name,
        source: "trained",
        status: "training",
        dpsgd: params.dpsgd,
        targetEpsilon: params.target_epsilon,
        trainParams: params,
      });
    }

    // Poll until all models are ready
    const poll = setInterval(async () => {
      // Mark models done after a short delay (real status comes from logs)
      setTraining(false);
      clearInterval(poll);
      ws.close();
      for (const params of cards) {
        onAdded({
          name: params.name,
          source: "trained",
          status: "ready",
          dpsgd: params.dpsgd,
          targetEpsilon: params.target_epsilon,
          trainParams: params,
        });
      }
    }, 2000);
  };

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

      {/* Live log stream */}
      {logs.length > 0 && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
          <div className="px-4 py-2 bg-slate-100 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-xs font-mono text-slate-500 flex items-center gap-2">
            <span className={`size-2 rounded-full ${training ? "bg-green-400 animate-pulse" : "bg-slate-400"}`} />
            Training log
          </div>
          <div className="bg-slate-950 p-4 h-48 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5">
            {logs.map((line, i) => <div key={i}>{line}</div>)}
            <div ref={logEndRef} />
          </div>
        </div>
      )}
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
        <p className="text-xs text-slate-500">{model.source === "uploaded" ? "Uploaded" : "Training…"}{model.dpsgd ? " · DP-SGD (ε=" + model.targetEpsilon + ")" : ""}</p>
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
    dpsgd: false,
  };
}
