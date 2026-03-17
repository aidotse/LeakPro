import React, { useState } from "react";

interface Props {
  label: string;
  hint: string;
  accept: string;
  icon: string;
  onFile: (file: File) => Promise<void>;
  onPath: (path: string) => Promise<void>;
}

export default function ServerOrUpload({ label, hint, accept, icon, onFile, onPath }: Props) {
  const [mode, setMode] = useState<"server" | "upload">("server");
  const [path, setPath] = useState("");
  const [done, setDone] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePath = async () => {
    if (!path.trim()) return;
    setLoading(true); setError(null);
    try {
      await onPath(path.trim());
      setDone(path.split("/").pop() ?? path);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setLoading(false); }
  };

  const handleFile = async (f: File) => {
    setLoading(true); setError(null);
    try {
      await onFile(f);
      setDone(f.name);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setLoading(false); }
  };

  if (done) {
    return (
      <div className="flex items-center gap-3 p-4 rounded-xl border border-green-500/30 bg-green-500/5">
        <span className="material-symbols-outlined text-green-500">check_circle</span>
        <div className="flex-1 min-w-0">
          <p className="font-bold text-sm text-green-600 dark:text-green-400 truncate">{done}</p>
          <p className="text-xs text-slate-500">{label}</p>
        </div>
        <button onClick={() => { setDone(null); setPath(""); }}
          className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200">
          <span className="material-symbols-outlined text-base">edit</span>
        </button>
      </div>
    );
  }

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-slate-200 dark:border-slate-700">
        {(["server", "upload"] as const).map((m) => (
          <button key={m} onClick={() => setMode(m)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 text-xs font-bold transition-colors
              ${mode === m ? "bg-primary/10 text-primary border-b-2 border-primary" : "text-slate-500 hover:bg-slate-50 dark:hover:bg-slate-800"}`}>
            <span className="material-symbols-outlined text-sm">{m === "server" ? "dns" : "upload_file"}</span>
            {m === "server" ? "Server path" : "Upload"}
          </button>
        ))}
      </div>

      <div className="p-4">
        <p className="text-xs font-semibold text-slate-500 mb-2">{label}</p>

        {mode === "server" ? (
          <div className="flex gap-2">
            <div className="flex-1 flex items-center gap-2 border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-2 bg-white dark:bg-slate-800 focus-within:border-primary transition-colors">
              <span className="material-symbols-outlined text-slate-400 text-sm shrink-0">folder_open</span>
              <input
                type="text" value={path}
                onChange={(e) => setPath(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handlePath()}
                placeholder="/absolute/path/on/server"
                className="flex-1 bg-transparent outline-none text-xs font-mono"
              />
            </div>
            <button onClick={handlePath} disabled={!path.trim() || loading}
              className="px-4 py-2 rounded-lg bg-primary text-white font-bold text-xs hover:bg-primary/90 disabled:opacity-50 shrink-0">
              {loading ? "…" : "Use"}
            </button>
          </div>
        ) : (
          <label className="flex items-center gap-3 border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-lg px-4 py-3 cursor-pointer hover:border-primary/50 transition-colors">
            <input type="file" className="hidden" accept={accept}
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
            <span className={`material-symbols-outlined text-2xl ${loading ? "text-primary animate-pulse" : "text-slate-400"}`}>{icon}</span>
            <div>
              <p className="text-sm font-semibold text-slate-600 dark:text-slate-300">{loading ? "Uploading…" : "Browse or drag file"}</p>
              <p className="text-xs text-slate-400">{hint}</p>
            </div>
          </label>
        )}

        {error && <p className="mt-2 text-xs text-red-500">{error}</p>}
      </div>
    </div>
  );
}
