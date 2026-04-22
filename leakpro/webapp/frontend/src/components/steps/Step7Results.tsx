import React, { useEffect, useState } from "react";
import { api, ModelResult, JobListItem } from "../../api";
import Summary from "../results/Summary";
import RocChart from "../results/RocChart";
import Histograms from "../results/Histograms";
import Records from "../results/Records";
import Venn from "../results/Venn";

const TABS = [
  { id: "summary",    label: "Summary",    icon: "dashboard" },
  { id: "roc",        label: "ROC Curves", icon: "show_chart" },
  { id: "histograms", label: "Histograms", icon: "bar_chart" },
  { id: "records",    label: "Records",    icon: "manage_search" },
  { id: "venn",       label: "Overlap",    icon: "workspaces" },
];

interface Props {
  jobId: string;
  onRestart: () => void;
}

export default function Step7Results({ jobId, onRestart }: Props) {
  const [allResults, setAllResults] = useState<ModelResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState("summary");

  // Compare panel state
  const [showCompare, setShowCompare] = useState(false);
  const [pastJobs, setPastJobs] = useState<JobListItem[] | null>(null);
  const [loadingJobId, setLoadingJobId] = useState<string | null>(null);
  const [loadedJobIds, setLoadedJobIds] = useState<Set<string>>(new Set([jobId]));

  useEffect(() => {
    api.getResults(jobId)
      .then((r) => setAllResults(r.results))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [jobId]);

  const openCompare = async () => {
    setShowCompare(true);
    if (!pastJobs) {
      try {
        const jobs = await api.listJobs();
        setPastJobs(jobs.filter((j) => j.status === "done"));
      } catch {
        setPastJobs([]);
      }
    }
  };

  const loadPastJob = async (pastJobId: string) => {
    if (loadedJobIds.has(pastJobId)) return;
    setLoadingJobId(pastJobId);
    try {
      const r = await api.getResults(pastJobId);
      setAllResults((prev) => {
        if (!prev) return r.results;
        const existing = new Set(prev.map((m) => `${m.job_id}/${m.model_name}`));
        const fresh = r.results.filter((m) => !existing.has(`${m.job_id}/${m.model_name}`));
        return [...prev, ...fresh];
      });
      setLoadedJobIds((prev) => new Set([...prev, pastJobId]));
    } catch {
      // ignore — job may have no results yet
    }
    setLoadingJobId(null);
  };

  const removeJob = (removeJobId: string) => {
    if (removeJobId === jobId) return; // can't remove current session
    setAllResults((prev) => prev?.filter((m) => m.job_id !== removeJobId) ?? prev);
    setLoadedJobIds((prev) => {
      const next = new Set(prev);
      next.delete(removeJobId);
      return next;
    });
  };

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

  if (!allResults) {
    return (
      <div className="flex flex-col items-center gap-4 py-16">
        <span className="material-symbols-outlined text-4xl text-primary animate-spin">sync</span>
        <p className="text-slate-500">Loading results…</p>
      </div>
    );
  }

  const sessionCount = loadedJobIds.size;

  return (
    <div className="flex flex-col gap-8">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-4xl font-black tracking-tight">Results</h2>
          <p className="text-slate-600 dark:text-slate-400 text-lg">
            {allResults.length} model{allResults.length !== 1 ? "s" : ""}
            {sessionCount > 1 && ` across ${sessionCount} sessions`}
          </p>
        </div>
        <div className="flex gap-2 shrink-0">
          <button
            onClick={openCompare}
            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-primary/50 text-primary text-sm font-bold hover:bg-primary/5 transition-colors"
          >
            <span className="material-symbols-outlined text-base">compare_arrows</span>
            Compare Sessions
          </button>
          <button
            onClick={onRestart}
            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-700 text-sm font-bold hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <span className="material-symbols-outlined text-base">restart_alt</span>
            New Audit
          </button>
        </div>
      </div>

      {/* Compare panel */}
      {showCompare && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 p-4 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <p className="font-bold text-sm">Load results from a previous session</p>
            <button onClick={() => setShowCompare(false)} className="material-symbols-outlined text-base text-slate-400 hover:text-slate-600">close</button>
          </div>
          {!pastJobs ? (
            <span className="material-symbols-outlined animate-spin text-primary">sync</span>
          ) : pastJobs.filter((j) => j.job_id !== jobId).length === 0 ? (
            <p className="text-slate-500 text-sm">No other completed sessions found.</p>
          ) : (
            <div className="flex flex-col gap-2">
              {pastJobs
                .filter((j) => j.job_id !== jobId)
                .map((j) => {
                  const loaded = loadedJobIds.has(j.job_id);
                  return (
                    <div key={j.job_id} className="flex items-center justify-between gap-3 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-3 py-2">
                      <div className="min-w-0">
                        <p className="text-xs font-mono text-slate-400 truncate">{j.job_id.slice(0, 12)}…</p>
                        <p className="text-sm text-slate-700 dark:text-slate-300 truncate">
                          {j.model_names.join(", ") || "no models"}
                        </p>
                        <p className="text-xs text-slate-400">{new Date(j.created_at).toLocaleString()}</p>
                      </div>
                      {loaded ? (
                        <div className="flex items-center gap-2 shrink-0">
                          <span className="text-xs text-green-600 font-semibold flex items-center gap-1">
                            <span className="material-symbols-outlined text-sm">check_circle</span> Loaded
                          </span>
                          <button
                            onClick={() => removeJob(j.job_id)}
                            className="material-symbols-outlined text-base text-slate-400 hover:text-red-500"
                            title="Remove from view"
                          >delete</button>
                        </div>
                      ) : (
                        <button
                          onClick={() => loadPastJob(j.job_id)}
                          disabled={loadingJobId === j.job_id}
                          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-primary text-white text-xs font-bold hover:bg-primary/90 disabled:opacity-50 shrink-0"
                        >
                          {loadingJobId === j.job_id
                            ? <span className="material-symbols-outlined text-sm animate-spin">sync</span>
                            : <span className="material-symbols-outlined text-sm">add</span>
                          }
                          Add
                        </button>
                      )}
                    </div>
                  );
                })}
            </div>
          )}
        </div>
      )}

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
        {tab === "summary"    && <Summary    results={allResults} />}
        {tab === "roc"        && <RocChart   results={allResults} />}
        {tab === "histograms" && <Histograms results={allResults} />}
        {tab === "records"    && <Records    results={allResults} jobId={jobId} />}
        {tab === "venn"       && <Venn       results={allResults} />}
      </div>
    </div>
  );
}
