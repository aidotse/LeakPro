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
  autoOpenCompare?: boolean;
}

interface PendingRename {
  pastJobId: string;
  results: ModelResult[];
  renames: Record<string, string>; // original model_name → display name
}

export default function Step7Results({ jobId, onRestart, autoOpenCompare }: Props) {
  const [allResults, setAllResults] = useState<ModelResult[] | null>(null);
  const [noResults, setNoResults] = useState(false);
  const [tab, setTab] = useState("summary");

  // Compare panel state
  const [showCompare, setShowCompare] = useState(autoOpenCompare ?? false);
  const [pastJobs, setPastJobs] = useState<JobListItem[] | null>(null);
  const [loadingJobId, setLoadingJobId] = useState<string | null>(null);
  const [loadedJobIds, setLoadedJobIds] = useState<Set<string>>(new Set([jobId]));

  // Rename dialog state
  const [pendingRename, setPendingRename] = useState<PendingRename | null>(null);

  useEffect(() => {
    api.getResults(jobId)
      .then((r) => {
        if (r.results.length === 0) setNoResults(true);
        setAllResults(r.results);
      })
      .catch(() => {
        setNoResults(true);
        setAllResults([]);
      });
  }, [jobId]);

  // Pre-load job list when arriving via "View Previous Results"
  useEffect(() => {
    if (autoOpenCompare && !pastJobs) {
      api.listJobs()
        .then((jobs) => setPastJobs(jobs.filter((j) => j.status === "done")))
        .catch(() => setPastJobs([]));
    }
  }, [autoOpenCompare]);

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

  const mergeResults = (pastJobId: string, newResults: ModelResult[], renames: Record<string, string>) => {
    const renamed = newResults.map((m) =>
      renames[m.model_name] ? { ...m, model_name: renames[m.model_name] } : m
    );
    setAllResults((prev) => {
      const base = prev ?? [];
      const existing = new Set(base.map((m) => `${m.job_id}/${m.model_name}`));
      const fresh = renamed.filter((m) => !existing.has(`${m.job_id}/${m.model_name}`));
      return [...base, ...fresh];
    });
    setNoResults(false);
    setLoadedJobIds((prev) => new Set([...prev, pastJobId]));
  };

  const loadPastJob = async (pastJobId: string) => {
    if (loadedJobIds.has(pastJobId)) return;
    setLoadingJobId(pastJobId);
    try {
      const r = await api.getResults(pastJobId);
      const existingNames = new Set((allResults ?? []).map((m) => m.model_name));
      const conflicts = r.results.filter((m) => existingNames.has(m.model_name));
      if (conflicts.length > 0) {
        const renames: Record<string, string> = {};
        conflicts.forEach((m) => { renames[m.model_name] = m.model_name + "_v2"; });
        setPendingRename({ pastJobId, results: r.results, renames });
      } else {
        mergeResults(pastJobId, r.results, {});
      }
    } catch {
      // ignore — job may have no results yet
    }
    setLoadingJobId(null);
  };

  const confirmRename = () => {
    if (!pendingRename) return;
    mergeResults(pendingRename.pastJobId, pendingRename.results, pendingRename.renames);
    setPendingRename(null);
  };

  const removeJob = (removeJobId: string) => {
    if (removeJobId === jobId) return;
    setAllResults((prev) => prev?.filter((m) => m.job_id !== removeJobId) ?? prev);
    setLoadedJobIds((prev) => {
      const next = new Set(prev);
      next.delete(removeJobId);
      return next;
    });
  };

  if (allResults === null) {
    return (
      <div className="flex flex-col items-center gap-4 py-16">
        <span className="material-symbols-outlined text-4xl text-primary animate-spin">sync</span>
        <p className="text-slate-500">Loading results…</p>
      </div>
    );
  }

  const sessionCount = loadedJobIds.size;
  const hasResults = allResults.length > 0;

  return (
    <div className="flex flex-col gap-8">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-4xl font-black tracking-tight">Results</h2>
          <p className="text-slate-600 dark:text-slate-400 text-lg">
            {hasResults
              ? `${allResults.length} model${allResults.length !== 1 ? "s" : ""}${sessionCount > 1 ? ` across ${sessionCount} sessions` : ""}`
              : "No results for this session yet"}
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
                  const attackSummary = j.model_names.map((name) => {
                    const atks = j.attacks_per_model?.[name] ?? [];
                    return atks.length ? `${name}: ${atks.join(", ")}` : name;
                  }).join(" · ");
                  return (
                    <div key={j.job_id} className="flex items-center justify-between gap-3 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-3 py-2">
                      <div className="min-w-0">
                        <p className="text-xs font-mono text-slate-400 truncate">{j.job_id.slice(0, 12)}…</p>
                        <p className="text-sm font-semibold text-slate-700 dark:text-slate-300 truncate">
                          {j.model_names.join(", ") || "no models"}
                        </p>
                        {attackSummary && (
                          <p className="text-xs text-slate-400 truncate">{attackSummary}</p>
                        )}
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

      {/* Rename modal */}
      {pendingRename && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setPendingRename(null)} />
          <div className="relative bg-white dark:bg-slate-900 rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-700 p-6 w-full max-w-md mx-4 flex flex-col gap-5">
            <div className="flex items-start gap-3">
              <span className="material-symbols-outlined text-orange-500 mt-0.5">edit</span>
              <div>
                <p className="font-bold text-slate-900 dark:text-slate-100">Model name conflict</p>
                <p className="text-sm text-slate-500 mt-0.5">
                  Some models share names with already-loaded ones. Rename them for display:
                </p>
              </div>
            </div>
            <div className="flex flex-col gap-3">
              {Object.entries(pendingRename.renames).map(([original, display]) => (
                <div key={original} className="flex items-center gap-3">
                  <span className="text-xs font-mono text-slate-400 w-28 shrink-0 truncate" title={original}>{original}</span>
                  <span className="material-symbols-outlined text-slate-300 text-sm">arrow_forward</span>
                  <input
                    value={display}
                    onChange={(e) => setPendingRename((prev) =>
                      prev ? { ...prev, renames: { ...prev.renames, [original]: e.target.value } } : prev
                    )}
                    className="flex-1 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm px-3 py-2 font-mono focus:outline-none focus:ring-2 focus:ring-primary/50"
                  />
                </div>
              ))}
            </div>
            <div className="flex gap-2 justify-end pt-1">
              <button
                onClick={() => setPendingRename(null)}
                className="px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-700 text-sm font-bold hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              >Cancel</button>
              <button
                onClick={confirmRename}
                disabled={Object.values(pendingRename.renames).some((v) => !v.trim())}
                className="px-4 py-2 rounded-lg bg-primary text-white text-sm font-bold hover:bg-primary/90 disabled:opacity-50 transition-colors"
              >Confirm &amp; Load</button>
            </div>
          </div>
        </div>
      )}

      {/* No results empty state */}
      {noResults && !hasResults && (
        <div className="flex flex-col items-center gap-3 py-12 text-center rounded-xl border border-dashed border-slate-300 dark:border-slate-700">
          <span className="material-symbols-outlined text-4xl text-slate-300 dark:text-slate-600">bar_chart</span>
          <p className="text-slate-500 font-semibold">No audit results for this session yet.</p>
          <p className="text-slate-400 text-sm">Load a previous session using "Compare Sessions" above, or run a new audit.</p>
        </div>
      )}

      {/* Tab bar + content — only when there are results */}
      {hasResults && (
        <>
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

          <div>
            {tab === "summary"    && <Summary    results={allResults} />}
            {tab === "roc"        && <RocChart   results={allResults} />}
            {tab === "histograms" && <Histograms results={allResults} />}
            {tab === "records"    && <Records    results={allResults} jobId={jobId} />}
            {tab === "venn"       && <Venn       results={allResults} />}
          </div>
        </>
      )}
    </div>
  );
}
