import React, { useEffect, useRef, useState } from "react";
import { api } from "../../api";
import { ModelEntry } from "./Step4Models";

interface Props {
  jobId: string;
  models: ModelEntry[];
  onDone: () => void;
}

export default function Step6Run({ jobId, models, onDone }: Props) {
  const [status, setStatus] = useState<"idle" | "running" | "done" | "failed">("idle");
  const [logs, setLogs] = useState<string[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const logEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const launch = async () => {
    setStatus("running");
    setLogs([]);
    setElapsed(0);

    timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);

    await api.startAudit(jobId);

    const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/jobs/${jobId}/logs`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      const msg: string = e.data;
      if (msg.startsWith("__STATUS__")) {
        const s = msg.replace("__STATUS__", "");
        setStatus(s === "done" ? "done" : "failed");
        if (timerRef.current) clearInterval(timerRef.current);
        ws.close();
        if (s === "done") setTimeout(onDone, 1500);
      } else {
        setLogs((prev) => [...prev, msg]);
      }
    };

    ws.onerror = () => {
      setStatus("failed");
      if (timerRef.current) clearInterval(timerRef.current);
    };
  };

  const fmtElapsed = (s: number) =>
    `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;

  return (
    <div className="flex flex-col gap-8">
      <div className="space-y-2">
        <h2 className="text-4xl font-black tracking-tight">Run Audit</h2>
        <p className="text-slate-600 dark:text-slate-400 text-lg max-w-2xl">
          {models.length} model{models.length !== 1 ? "s" : ""} queued. The audit runs in the
          background — you can watch the live log below.
        </p>
      </div>

      {/* Summary cards */}
      <div className="flex gap-4 flex-wrap">
        {models.map((m) => (
          <div key={m.name} className="flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50 text-sm">
            <span className="material-symbols-outlined text-primary text-base">psychology</span>
            <span className="font-semibold">{m.name}</span>
            {m.dpsgd && <span className="text-xs text-blue-500 font-semibold">DP-SGD</span>}
          </div>
        ))}
      </div>

      {/* Launch button */}
      {status === "idle" && (
        <button
          onClick={launch}
          className="self-start px-8 py-3 rounded-lg bg-primary text-white font-bold text-lg hover:bg-primary/90 transition-colors flex items-center gap-3 shadow-xl shadow-primary/20"
        >
          <span className="material-symbols-outlined">rocket_launch</span>
          Launch Audit
        </button>
      )}

      {/* Running state */}
      {status === "running" && (
        <div className="flex items-center gap-3 text-slate-600 dark:text-slate-400">
          <span className="material-symbols-outlined text-primary animate-spin">sync</span>
          <span className="font-semibold">Running…</span>
          <span className="font-mono text-sm">{fmtElapsed(elapsed)}</span>
        </div>
      )}

      {/* Done */}
      {status === "done" && (
        <div className="flex items-center gap-3 text-green-600 dark:text-green-400 font-bold">
          <span className="material-symbols-outlined">check_circle</span>
          Audit complete — loading results…
        </div>
      )}

      {/* Failed */}
      {status === "failed" && (
        <div className="flex items-center gap-3 text-red-500 font-bold">
          <span className="material-symbols-outlined">error</span>
          Audit failed. Check logs below.
        </div>
      )}

      {/* Live log terminal — always visible once launched */}
      {status !== "idle" && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
          <div className="px-4 py-2 bg-slate-100 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-xs font-mono text-slate-500 flex items-center gap-2">
            <span className={`size-2.5 rounded-full ${status === "running" ? "bg-green-400 animate-pulse" : status === "done" ? "bg-green-400" : "bg-red-400"}`} />
            Live output
            {status === "running" && <span className="ml-auto font-mono">{fmtElapsed(elapsed)}</span>}
          </div>
          <div className="bg-slate-950 p-4 h-72 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5">
            {logs.length === 0 && (
              <span className="text-slate-600 animate-pulse">Waiting for output…</span>
            )}
            {logs.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}
