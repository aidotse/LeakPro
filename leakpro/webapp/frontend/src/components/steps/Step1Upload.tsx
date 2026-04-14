import React, { useCallback, useRef, useState } from "react";
import ReactDOM from "react-dom";
import { api, DataMeta } from "../../api";

// ---------------------------------------------------------------------------
// dataset_handler.py template
// ---------------------------------------------------------------------------
const DATASET_HANDLER_TEMPLATE = `"""
dataset_handler.py — define how LeakPro loads and normalises YOUR data.
Upload this file in Step 1 (Dataset).
"""
import numpy as np
import torch
from leakpro import AbstractInputHandler


class UserDataset(AbstractInputHandler.UserDataset):
    """
    Wraps your dataset for use in LeakPro.

    Requirements:
      - Must be indexable: dataset[i] returns (input_tensor, label)
      - data: float32 Tensor in [0, 1], shape (N, C, H, W) for images
              or (N, features) for tabular
      - targets: integer class labels, shape (N,)
    """
    def __init__(self, data, targets, **kwargs):
        # Convert numpy arrays to float32 tensors in [0, 1]
        if isinstance(data, np.ndarray):
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0   # uint8 -> float [0,1]
            else:
                data = data.astype(np.float32)
            # Channel-last (N,H,W,C) -> channel-first (N,C,H,W) for images
            if data.ndim == 4 and data.shape[-1] in (1, 3, 4) and data.shape[1] > 4:
                data = data.transpose(0, 3, 1, 2)
            data = torch.from_numpy(data)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).long()

        assert data.max() <= 1.0 and data.min() >= 0.0, "Data must be in [0, 1] after conversion"

        self.data = data.float()
        self.targets = targets

        # Per-channel normalisation stats (computed from your data)
        if data.ndim == 4:
            self.mean = self.data.mean(dim=(0, 2, 3)).view(-1, 1, 1)
            self.std  = self.data.std(dim=(0, 2, 3)).view(-1, 1, 1).clamp(min=1e-7)
        else:
            self.mean = self.data.mean(dim=0)
            self.std  = self.data.std(dim=0).clamp(min=1e-7)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def __getitem__(self, index):
        # Must be indexable — returns (normalised_input, label)
        return self._normalize(self.data[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)
`;

function downloadTemplate() {
  const blob = new Blob([DATASET_HANDLER_TEMPLATE], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "dataset_handler.py";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Code preview modal (same pattern as Step3Setup)
// ---------------------------------------------------------------------------
function CodeModal({ onClose }: { onClose: () => void }) {
  return ReactDOM.createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-slate-900 rounded-2xl shadow-2xl w-full max-w-3xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-800">
          <h3 className="font-bold text-lg">Example: Data Handler</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>
        <p className="px-6 pt-4 text-sm text-slate-500 dark:text-slate-400">
          Your <code>dataset_handler.py</code> must define a <code>UserDataset</code> class that is
          <strong> indexable</strong> — <code>dataset[i]</code> must return a <code>(input, label)</code> pair.
          Data must be a float32 tensor in [0, 1]. Adapt the normalisation and channel order to match your data format.
        </p>
        <pre className="overflow-auto px-6 py-4 font-mono text-xs text-slate-300 bg-slate-950 mx-6 my-4 rounded-xl max-h-[55vh] leading-relaxed">
          {DATASET_HANDLER_TEMPLATE}
        </pre>
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-slate-200 dark:border-slate-800">
          <button
            onClick={downloadTemplate}
            className="flex items-center gap-2 px-5 py-2 rounded-lg border border-slate-300 dark:border-slate-700 text-sm font-bold hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <span className="material-symbols-outlined text-base">download</span>
            Download template
          </button>
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-lg bg-primary text-white text-sm font-bold hover:bg-primary/90 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}

// ---------------------------------------------------------------------------
// DatasetHandlerUpload — shown after dataset is validated, required to continue
// ---------------------------------------------------------------------------
function DatasetHandlerUpload({ jobId, onUploaded }: { jobId: string; onUploaded: (uploaded: boolean) => void }) {
  const [handlerFile, setHandlerFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (f: File) => {
    setUploading(true);
    setError(null);
    try {
      await api.uploadDatasetHandler(jobId, f);
      setHandlerFile(f);
      onUploaded(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  };

  const remove = () => {
    setHandlerFile(null);
    onUploaded(false);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="flex flex-col gap-3 p-5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50">
      {showModal && <CodeModal onClose={() => setShowModal(false)} />}

      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-bold text-sm">Data Handler</p>
          <p className="text-xs text-slate-500 mt-0.5">
            Define how LeakPro wraps your data for training and auditing. Must define a{" "}
            <code className="text-xs bg-slate-200 dark:bg-slate-700 px-1 rounded">UserDataset</code> class
            that is <strong>indexable</strong> — <code className="text-xs bg-slate-200 dark:bg-slate-700 px-1 rounded">dataset[i]</code> returns
            an <code className="text-xs bg-slate-200 dark:bg-slate-700 px-1 rounded">(input, label)</code> pair.
            Adapt normalisation and format to your data type (images, tabular, time series, etc.).
          </p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="shrink-0 flex items-center gap-1 text-xs text-primary font-semibold hover:underline"
        >
          <span className="material-symbols-outlined text-sm">code</span>
          View example
        </button>
      </div>

      <div className="flex items-center gap-3">
        <input
          ref={inputRef}
          type="file"
          accept=".py"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        {handlerFile ? (
          <div className="flex items-center gap-2 flex-1 px-4 py-2 rounded-lg border border-green-500/40 bg-green-500/5 text-green-600 dark:text-green-400 text-sm font-semibold">
            <span className="material-symbols-outlined text-base">check_circle</span>
            {handlerFile.name}
          </div>
        ) : (
          <button
            onClick={() => inputRef.current?.click()}
            disabled={uploading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-dashed border-slate-300 dark:border-slate-700 text-sm text-slate-500 hover:border-primary hover:text-primary transition-colors disabled:opacity-50"
          >
            <span className="material-symbols-outlined text-base">upload_file</span>
            {uploading ? "Uploading…" : "Upload dataset_handler.py"}
          </button>
        )}
        {handlerFile && (
          <button onClick={remove} className="text-xs text-slate-400 hover:text-red-500 transition-colors">
            Remove
          </button>
        )}
      </div>
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  );
}

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
  const [handlerUploaded, setHandlerUploaded] = useState(false);

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
        <>
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
          </div>
          <DatasetHandlerUpload jobId={jobId} onUploaded={setHandlerUploaded} />
          <div className="flex justify-end">
            <button
              onClick={() => onDone(meta)}
              disabled={!handlerUploaded}
              className="px-8 py-2.5 rounded-lg bg-primary text-white font-bold hover:bg-primary/90 transition-colors flex items-center gap-2 shadow-lg shadow-primary/20 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Continue <span className="material-symbols-outlined text-base">arrow_forward</span>
            </button>
          </div>
        </>
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
  const [handlerUploaded, setHandlerUploaded] = useState(false);

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
      <>
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
        </div>
        <DatasetHandlerUpload jobId={jobId} onUploaded={setHandlerUploaded} />
        <div className="flex justify-end">
          <button
            onClick={() => onDone(meta)}
            disabled={!handlerUploaded}
            className="px-8 py-2.5 rounded-lg bg-primary text-white font-bold hover:bg-primary/90 transition-colors flex items-center gap-2 shadow-lg shadow-primary/20 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Continue <span className="material-symbols-outlined text-base">arrow_forward</span>
          </button>
        </div>
      </>
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
