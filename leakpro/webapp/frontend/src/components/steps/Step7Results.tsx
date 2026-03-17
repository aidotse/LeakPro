import React, { useEffect, useState } from "react";
import { api, ModelResult } from "../../api";
import Summary from "../results/Summary";
import RocChart from "../results/RocChart";
import Histograms from "../results/Histograms";
import Records from "../results/Records";

const TABS = [
  { id: "summary",    label: "Summary",    icon: "dashboard" },
  { id: "roc",        label: "ROC Curves", icon: "show_chart" },
  { id: "histograms", label: "Histograms", icon: "bar_chart" },
  { id: "records",    label: "Records",    icon: "manage_search" },
];

interface Props {
  jobId: string;
  onRestart: () => void;
}

export default function Step7Results({ jobId, onRestart }: Props) {
  const [results, setResults] = useState<ModelResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState("summary");

  useEffect(() => {
    api.getResults(jobId)
      .then((r) => setResults(r.results))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [jobId]);

  if (error) {
    return (
      <div className="flex flex-col items-center gap-4 py-16 text-center">
        <span className="material-symbols-outlined text-5xl text-red-500">error</span>
        <p className="text-red-500 font-bold">{error}</p>
        <button onClick={onRestart} className="text-primary hover:underline text-sm font-semibold">
          Start a new audit
        </button>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="flex flex-col items-center gap-4 py-16">
        <span className="material-symbols-outlined text-4xl text-primary animate-spin">sync</span>
        <p className="text-slate-500">Loading results…</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-4xl font-black tracking-tight">Results</h2>
          <p className="text-slate-600 dark:text-slate-400 text-lg">
            {results.length} model{results.length !== 1 ? "s" : ""} audited
          </p>
        </div>
        <button
          onClick={onRestart}
          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-700 text-sm font-bold hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors shrink-0"
        >
          <span className="material-symbols-outlined text-base">restart_alt</span>
          New Audit
        </button>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-slate-200 dark:border-slate-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-bold border-b-2 transition-colors -mb-px
              ${tab === t.id
                ? "border-primary text-primary"
                : "border-transparent text-slate-500 hover:text-slate-900 dark:hover:text-slate-100"
              }`}
          >
            <span className="material-symbols-outlined text-base">{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>
        {tab === "summary"    && <Summary    results={results} />}
        {tab === "roc"        && <RocChart   results={results} />}
        {tab === "histograms" && <Histograms results={results} />}
        {tab === "records"    && <Records    results={results} />}
      </div>
    </div>
  );
}
