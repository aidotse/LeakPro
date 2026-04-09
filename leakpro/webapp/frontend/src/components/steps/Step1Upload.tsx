import React, { useCallback, useState } from "react";
import { api, DataMeta } from "../../api";

interface Props {
  jobId: string;
  onDone: (meta: DataMeta) => void;
}

type Mode = "server" | "upload";

export default function Step1Upload({ jobId, onDone }: Props) {
  const [mode, setMode] = useState<Mode>("server");

  return (
    <div className="flex flex-col gap-8">
      <div className="space-y-2">
        <h2 className="text-4xl font-black tracking-tight">Load Dataset</h2>
        <p className="text-slate-600 dark:text-slate-400 text-lg max-w-2xl">
          Point to your training dataset on the server, or upload a file from your local machine.
          Supports CSV, JSONL, Parquet, NumPy (.npy), or PyTorch (.pkl / .pt).
        </p>
      </div>

      {/* Mode toggle */}
      <div className="flex gap-2">
        <ModeButton active={mode === "server"} onClick={() => setMode("server")}
          icon="dns" label="Server path" />
        <ModeButton active={mode === "upload"} onClick={() => setMode("upload")}
          icon="upload_file" label="Upload file" />
      </div>

      {mode === "server"
        ? <ServerPathForm jobId={jobId} onDone={onDone} />
        : <UploadForm jobId={jobId} onDone={onDone} />
      }
    </div>
  );
}

// ---------------------------------------------------------------------------
// Server path form
// ---------------------------------------------------------------------------
function ServerPathForm({ jobId, onDone }: { jobId: string; onDone: (m: DataMeta) => void }) {
  const [path, setPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<DataMeta | null>(null);

  const validate = async () => {
    if (!path.trim()) return;
    setLoading(true);
    setError(null);
    setMeta(null);
    try {
      const m = await api.setDataPath(jobId, path.trim());
      setMeta(m);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <label className="text-xs font-semibold text-slate-500">Absolute path on server</label>
        <div className="flex gap-2">
          <div className="flex-1 flex items-center gap-2 border border-slate-300 dark:border-slate-700 rounded-lg px-4 py-3 bg-white dark:bg-slate-800 focus-within:border-primary transition-colors">
            <span className="material-symbols-outlined text-slate-400 text-base shrink-0">folder_open</span>
            <input
              type="text"
              value={path}
              onChange={(e) => { setPath(e.target.value); setMeta(null); setError(null); }}
              onKeyDown={(e) => e.key === "Enter" && validate()}
              placeholder="/home/fazeleh/data/cifar_train.pkl"
              className="flex-1 bg-transparent outline-none text-sm font-mono"
            />
          </div>
          <button
            onClick={validate}
            disabled={!path.trim() || loading}
            className="px-5 py-3 rounded-lg bg-primary text-white font-bold text-sm hover:bg-primary/90 transition-colors disabled:opacity-50 shrink-0"
          >
            {loading ? "Checking…" : "Validate"}
          </button>
        </div>
        <p className="text-xs text-slate-400">The file must be accessible to the LeakPro server process.</p>
      </div>

      {error && (
        <div className="flex items-start gap-2 p-4 rounded-lg border border-red-500/30 bg-red-500/5 text-red-500 text-sm">
          <span className="material-symbols-outlined text-base shrink-0">error</span>
          {error}
        </div>
      )}

      {meta && (
        <div className="flex flex-col gap-4 p-5 rounded-xl border border-green-500/30 bg-green-500/5">
          <div className="flex items-center gap-2 text-green-600 dark:text-green-400 font-bold">
            <span className="material-symbols-outlined">check_circle</span>
            Dataset found and analysed
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetaField label="Type"    value={meta.data_type} />
            <MetaField label="Shape"   value={`[${meta.shape.join(", ")}]`} />
            <MetaField label="Samples" value={meta.n_samples.toLocaleString()} />
            <MetaField label="Dtype"   value={meta.dtype} />
          </div>
          <button
            onClick={() => onDone(meta)}
            className="self-end px-8 py-2.5 rounded-lg bg-primary text-white font-bold hover:bg-primary/90 transition-colors flex items-center gap-2 shadow-lg shadow-primary/20"
          >
            Continue <span className="material-symbols-outlined text-base">arrow_forward</span>
          </button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload form (original drag-drop)
// ---------------------------------------------------------------------------
function UploadForm({ jobId, onDone }: { jobId: string; onDone: (m: DataMeta) => void }) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<DataMeta | null>(null);

  const handleFile = useCallback(async (f: File) => {
    setFile(f);
    setError(null);
    setMeta(null);
    setLoading(true);
    try {
      const result = await api.uploadData(jobId, f);
      setMeta(result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  return (
    <div className="flex flex-col gap-6">
    <label
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-12 cursor-pointer transition-all
        ${dragging ? "border-primary bg-primary/10"
          : file ? "border-green-500 bg-green-500/5"
          : "border-slate-300 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50 hover:border-primary/50 hover:bg-primary/5"
        }`}
    >
      <input type="file" className="hidden"
        accept=".csv,.jsonl,.json,.parquet,.npy,.pkl,.pt,.pth"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
      <div className="size-16 rounded-full bg-primary/10 flex items-center justify-center text-primary mb-6">
        <span className="material-symbols-outlined text-3xl">{file ? "check_circle" : "upload_file"}</span>
      </div>
      <div className="text-center mb-6">
        {file ? (
          <>
            <h3 className="text-xl font-bold mb-1 text-green-600 dark:text-green-400">{file.name}</h3>
            <p className="text-slate-500">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
          </>
        ) : (
          <>
            <h3 className="text-xl font-bold mb-2">Drag and drop your file here</h3>
            <p className="text-slate-500 dark:text-slate-400">CSV, JSONL, Parquet, .npy, .pkl, .pt — up to 2 GB</p>
          </>
        )}
      </div>
      {!file && (
        <span className="bg-primary text-white px-6 py-2.5 rounded-lg font-bold hover:bg-primary/90 transition-colors shadow-lg shadow-primary/20">
          Browse Files
        </span>
      )}
      {loading && <p className="mt-4 text-sm text-slate-500 animate-pulse">Analysing dataset…</p>}
      {error && <p className="mt-4 text-sm text-red-500">{error}</p>}
    </label>

    {meta && (
      <div className="flex flex-col gap-4 p-5 rounded-xl border border-green-500/30 bg-green-500/5">
        <div className="flex items-center gap-2 text-green-600 dark:text-green-400 font-bold">
          <span className="material-symbols-outlined">check_circle</span>
          Dataset analysed
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetaField label="Type"    value={meta.data_type} />
          <MetaField label="Shape"   value={`[${meta.shape.join(", ")}]`} />
          <MetaField label="Samples" value={meta.n_samples.toLocaleString()} />
          <MetaField label="Dtype"   value={meta.dtype} />
        </div>
        <button
          onClick={() => onDone(meta)}
          className="self-end px-8 py-2.5 rounded-lg bg-primary text-white font-bold hover:bg-primary/90 transition-colors flex items-center gap-2 shadow-lg shadow-primary/20"
        >
          Continue <span className="material-symbols-outlined text-base">arrow_forward</span>
        </button>
      </div>
    )}
  </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function ModeButton({ active, onClick, icon, label }: {
  active: boolean; onClick: () => void; icon: string; label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-bold text-sm transition-colors
        ${active
          ? "bg-primary text-white shadow-lg shadow-primary/20"
          : "border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
        }`}
    >
      <span className="material-symbols-outlined text-base">{icon}</span>
      {label}
    </button>
  );
}

function MetaField({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      <p className="font-mono text-sm font-bold">{value}</p>
    </div>
  );
}
