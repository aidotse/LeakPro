import React, { useState } from "react";
import { api, ArchConfig, HandlerConfig } from "../../api";
import ServerOrUpload from "../ServerOrUpload";

interface Props {
  jobId: string;
  handlerConfig: HandlerConfig;
  onDone: (arch: ArchConfig) => void;
}

const PRESETS: Array<{ id: string; label: string; icon: string; desc: string; types: string[] }> = [
  {
    id: "cifar_image",
    label: "Image (CIFAR-style)",
    icon: "image",
    desc: "WideResNet architecture with standard image augmentation. Works for 3-channel images (32×32 or larger).",
    types: ["image"],
  },
  {
    id: "tabular_mlp",
    label: "Tabular (MLP)",
    icon: "table_rows",
    desc: "Multi-layer perceptron for tabular / CSV data with numeric features.",
    types: ["tabular"],
  },
  {
    id: "time_series",
    label: "Time Series (GRU)",
    icon: "show_chart",
    desc: "GRU-based sequence model for time-series data.",
    types: ["time_series"],
  },
];

export default function Step3Setup({ jobId, handlerConfig, onDone }: Props) {
  const [mode, setMode] = useState<"preset" | "upload">("preset");
  const [selectedPreset, setSelectedPreset] = useState<string | null>(
    PRESETS.find((p) => p.types.includes(handlerConfig.data_type))?.id ?? null
  );
  const [archFile, setArchFile] = useState<File | null>(null);
  const [handlerFile, setHandlerFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canProceed =
    mode === "preset" ? !!selectedPreset : !!(archFile && handlerFile);

  const proceed = async () => {
    setLoading(true);
    setError(null);
    try {
      if (mode === "upload" && archFile && handlerFile) {
        await api.uploadArch(jobId, archFile);
        await api.uploadHandler(jobId, handlerFile);
      }
      const config: ArchConfig = {
        preset: mode === "preset" ? selectedPreset! : undefined,
        arch_filename: archFile?.name,
        handler_filename: handlerFile?.name,
      };
      await api.setArchConfig(jobId, config);
      onDone(config);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-8">
      <div className="space-y-2">
        <h2 className="text-4xl font-black tracking-tight">Architecture & Training</h2>
        <p className="text-slate-600 dark:text-slate-400 text-lg max-w-2xl">
          Choose a built-in preset that matches your data type, or upload your own architecture
          and training loop.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="flex gap-2">
        {(["preset", "upload"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-5 py-2 rounded-lg font-bold text-sm transition-colors
              ${mode === m
                ? "bg-primary text-white shadow-lg shadow-primary/20"
                : "border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
              }`}
          >
            {m === "preset" ? "Use built-in preset" : "Upload my own"}
          </button>
        ))}
      </div>

      {mode === "preset" ? (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {PRESETS.map((p) => {
            const recommended = p.types.includes(handlerConfig.data_type);
            return (
              <button
                key={p.id}
                onClick={() => setSelectedPreset(p.id)}
                className={`p-5 rounded-xl border text-left transition-all
                  ${selectedPreset === p.id
                    ? "border-primary bg-primary/5 shadow-lg shadow-primary/10"
                    : "border-slate-200 dark:border-slate-800 hover:border-primary/40"
                  }`}
              >
                <div className="flex items-start gap-3 mb-3">
                  <div className="size-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary shrink-0">
                    <span className="material-symbols-outlined">{p.icon}</span>
                  </div>
                  <div>
                    <p className="font-bold text-sm">{p.label}</p>
                    {recommended && (
                      <span className="text-xs text-green-600 dark:text-green-400 font-semibold">
                        ✓ Recommended for your data
                      </span>
                    )}
                  </div>
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400">{p.desc}</p>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <ServerOrUpload
            label="Model Architecture (.py)"
            hint="Python file defining your nn.Module subclass"
            icon="code"
            accept=".py"
            onFile={async (f) => { await api.uploadArch(jobId, f); setArchFile(f); }}
            onPath={async (p) => { await api.setArchPath(jobId, p); setArchFile(new File([], p.split("/").pop() ?? "arch.py")); }}
          />
          <ServerOrUpload
            label="Training Loop / Handler (.py)"
            hint="Python file with your training handler class"
            icon="settings"
            accept=".py"
            onFile={async (f) => { await api.uploadHandler(jobId, f); setHandlerFile(f); }}
            onPath={async (p) => { await api.setHandlerPath(jobId, p); setHandlerFile(new File([], p.split("/").pop() ?? "handler.py")); }}
          />
        </div>
      )}

      {error && <p className="text-sm text-red-500">{error}</p>}

      <div className="flex justify-end">
        <button
          onClick={proceed}
          disabled={!canProceed || loading}
          className="px-8 py-2.5 rounded-lg bg-primary text-white font-bold hover:bg-primary/90 transition-colors flex items-center gap-2 shadow-lg shadow-primary/20 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Saving…" : "Continue"}
          <span className="material-symbols-outlined text-base">arrow_forward</span>
        </button>
      </div>
    </div>
  );
}

function FileSlot({
  label, hint, icon, file, onChange, accept,
}: {
  label: string; hint: string; icon: string;
  file: File | null; onChange: (f: File) => void; accept: string;
}) {
  return (
    <label className={`flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-8 cursor-pointer transition-all
      ${file
        ? "border-green-500 bg-green-500/5"
        : "border-slate-300 dark:border-slate-800 hover:border-primary/50 hover:bg-primary/5"
      }`}
    >
      <input type="file" className="hidden" accept={accept}
        onChange={(e) => e.target.files?.[0] && onChange(e.target.files[0])} />
      <span className={`material-symbols-outlined text-3xl mb-3 ${file ? "text-green-500" : "text-primary"}`}>
        {file ? "check_circle" : icon}
      </span>
      <p className="font-bold text-sm text-center mb-1">{file ? file.name : label}</p>
      <p className="text-xs text-slate-500 text-center">{hint}</p>
    </label>
  );
}
