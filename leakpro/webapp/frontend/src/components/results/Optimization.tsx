import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { api, ModelResult, OptimizationRun, Setting, Verification } from "../../api";
import { COPY, tagsFor } from "./optimizationCopy";

const ACCENT = "#193ce6";
const TESTED = "#94a3b8";
const POLL_MS = 2000;

/** Relative worsening of attack success that makes an estimate "optimistic". */
const OPTIMISTIC_MARGIN = 0.2;

interface Props {
  jobId: string;
  model: ModelResult;
  onBack: () => void;
  onAdopted: (setting: Setting, verification: Verification) => void;
}

const pct = (v: number | undefined, digits = 1) =>
  v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`;

/**
 * Reduce the run's best settings to the 3-5 the view labels.
 *
 * The run returns however many are non-dominated, which can be one or twenty.
 * Keep both ends of the trade-off and spread the rest evenly between them, so
 * the labels always describe genuinely different choices.
 */
function pickHighlights(best: Setting[], max = 5): Setting[] {
  if (best.length <= max) return best;
  const step = (best.length - 1) / (max - 1);
  return Array.from({ length: max }, (_, i) => best[Math.round(i * step)]);
}

export default function Optimization({ jobId, model, onBack, onAdopted }: Props) {
  const [run, setRun] = useState<OptimizationRun | null>(null);
  const [maxQualityLoss, setMaxQualityLoss] = useState(0.05);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [advanced, setAdvanced] = useState<Record<string, number>>({});
  const [selected, setSelected] = useState<Setting | null>(null);
  const [verification, setVerification] = useState<Verification | null>(null);
  const [adopted, setAdopted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const phase: "setup" | "running" | "done" =
    run == null || run.status === "idle" ? "setup" : run.status === "running" ? "running" : "done";

  // ── polling ──────────────────────────────────────────────────────────────
  // `silent` covers the first look: a model that has never been optimized has
  // no run to fetch, which is the normal starting state, not a failure.
  const poll = useCallback(async (silent = false) => {
    try {
      setRun(await api.getOptimization(jobId, model.model_name));
    } catch (e) {
      if (!silent) setError(e instanceof Error ? e.message : String(e));
    }
  }, [jobId, model.model_name]);

  useEffect(() => { poll(true); }, [poll]);

  useEffect(() => {
    if (run?.status !== "running") return;
    pollRef.current = setInterval(() => poll(true), POLL_MS);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [run?.status, poll]);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const start = async () => {
    setStarting(true);
    setError(null);
    try {
      await api.startOptimization(jobId, model.model_name, { max_quality_loss: maxQualityLoss, advanced });
      await poll();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  // ── derived ──────────────────────────────────────────────────────────────
  const baseline = run?.baseline_utility ?? model.test_accuracy;
  const qualityFloor = baseline !== undefined ? baseline * (1 - maxQualityLoss) : undefined;

  const tested = useMemo(
    () => (run?.settings ?? []).filter((s) => s.attack_tpr != null),
    [run?.settings],
  );

  const highlights = useMemo(() => {
    if (!run) return [];
    const byIndex = new Map(run.settings.map((s) => [s.index, s]));
    // Sorted safest-first so the tags line up with what they claim.
    const best = run.best_indices
      .map((i) => byIndex.get(i))
      .filter((s): s is Setting => s != null && s.attack_tpr != null)
      .sort((a, b) => a.attack_tpr! - b.attack_tpr!);
    return pickHighlights(best);
  }, [run]);

  const tags = tagsFor(highlights.length);
  const withinLimit = (s: Setting) => qualityFloor === undefined || s.utility >= qualityFloor;

  // ── verification ─────────────────────────────────────────────────────────
  const confirm = async (setting: Setting) => {
    setAdopted(false);
    setError(null);
    try {
      setVerification(await api.verifySetting(jobId, model.model_name, setting.index));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    if (verification?.status !== "running" || !selected) return;
    const id = setInterval(async () => {
      try {
        setVerification(await api.getVerification(jobId, model.model_name, selected.index));
      } catch { /* keep the last state; the next tick retries */ }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [verification?.status, selected, jobId, model.model_name]);

  const adopt = async () => {
    if (!selected || !verification) return;
    try {
      await api.adoptSetting(jobId, model.model_name, selected.index);
      setAdopted(true);
      onAdopted(selected, verification);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const optimistic =
    verification?.verified != null &&
    verification.estimated.attack_tpr > 0 &&
    verification.verified.attack_tpr > verification.estimated.attack_tpr * (1 + OPTIMISTIC_MARGIN);

  // ── chart ────────────────────────────────────────────────────────────────
  const highlightIndices = new Set(highlights.map((s) => s.index));
  const backdrop = phase === "done" ? tested.filter((s) => !highlightIndices.has(s.index)) : tested;

  const traces: Plotly.Data[] = [
    {
      x: backdrop.map((s) => s.attack_tpr! * 100),
      y: backdrop.map((s) => s.utility * 100),
      mode: "markers",
      type: "scatter",
      name: "Tested",
      marker: {
        size: 9,
        color: phase === "done" ? TESTED : ACCENT,
        opacity: phase === "done" ? 0.25 : 0.75,
        line: { width: 0 },
      },
      hovertemplate: "Attack success %{x:.2f}%<br>Quality %{y:.2f}%<extra></extra>",
    },
  ];

  if (phase === "done" && highlights.length > 0) {
    traces.push({
      x: highlights.map((s) => s.attack_tpr! * 100),
      y: highlights.map((s) => s.utility * 100),
      mode: "lines",
      type: "scatter",
      name: COPY.resultsTitle,
      line: { color: ACCENT, width: 2 },
      hoverinfo: "skip",
      showlegend: false,
    });
    traces.push({
      x: highlights.map((s) => s.attack_tpr! * 100),
      y: highlights.map((s) => s.utility * 100),
      text: highlights.map((_, i) => tags[i] ?? ""),
      mode: "text+markers",
      type: "scatter",
      name: COPY.resultsTitle,
      textposition: "top right",
      textfont: { size: 12, color: ACCENT },
      marker: {
        size: 15,
        color: ACCENT,
        // plotly.js accepts a per-point opacity array here; its typings do not.
        opacity: highlights.map((s) => (withinLimit(s) ? 1 : 0.3)) as unknown as number,
        line: { width: 2, color: "#ffffff" },
      },
      customdata: highlights.map((s) => (withinLimit(s) ? "" : `<br><b>${COPY.exceedsLimit}</b>`)),
      hovertemplate:
        "Attack success %{x:.2f}%<br>Quality %{y:.2f}%%{customdata}<extra></extra>",
    });
  }

  const shapes: Partial<Plotly.Shape>[] =
    qualityFloor !== undefined && tested.length > 0
      ? [{
          type: "line", xref: "paper", x0: 0, x1: 1,
          yref: "y", y0: qualityFloor * 100, y1: qualityFloor * 100,
          line: { color: TESTED, width: 1, dash: "dash" },
        }]
      : [];

  // ── render ───────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-8">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-3xl font-black tracking-tight">{COPY.title}</h2>
          <p className="text-slate-600 dark:text-slate-300 max-w-2xl">{COPY.intro}</p>
          <p className="text-sm text-slate-400">
            For <span className="font-semibold text-slate-600 dark:text-slate-300">{model.model_name}</span>
          </p>
        </div>
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-700 text-sm font-bold hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors shrink-0"
        >
          <span className="material-symbols-outlined text-base">arrow_back</span>
          {COPY.back}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-500">
          {error}
        </div>
      )}

      {/* Constraint — one control, everything else behind Advanced */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 p-5 flex flex-col gap-4">
        <div className="flex items-center gap-6 flex-wrap">
          <div className="flex-1 min-w-[280px]">
            <label className="text-sm font-bold block mb-2">
              {COPY.constraintLabel}:{" "}
              <span className="text-primary font-mono">{(maxQualityLoss * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range" min={0} max={0.5} step={0.01}
              value={maxQualityLoss}
              onChange={(e) => setMaxQualityLoss(Number(e.target.value))}
              className="w-full accent-primary"
            />
            <p className="text-xs text-slate-400 mt-1">{COPY.constraintHelp}</p>
          </div>
          <div className="flex gap-6 text-sm">
            <div>
              <p className="text-xs text-slate-400">{COPY.baselineLabel}</p>
              <p className="font-mono font-bold text-lg">{pct(baseline)}</p>
            </div>
            <div>
              <p className="text-xs text-slate-400">{COPY.floorLabel}</p>
              <p className="font-mono font-bold text-lg">{pct(qualityFloor)}</p>
            </div>
          </div>
        </div>

        <div>
          <button
            onClick={() => setAdvancedOpen(!advancedOpen)}
            className="text-xs font-bold text-slate-500 hover:text-primary flex items-center gap-1 transition-colors"
          >
            <span className="material-symbols-outlined text-sm">
              {advancedOpen ? "expand_less" : "expand_more"}
            </span>
            {COPY.advanced}
          </button>
          {advancedOpen && (
            <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
              <p className="col-span-full text-xs text-slate-400">{COPY.advancedHelp}</p>
              {(["noise_multiplier", "max_grad_norm", "learning_rate", "batch_size"] as const).map((k) => (
                <div key={k}>
                  <label className="text-xs font-semibold text-slate-500 mb-1 block">{k}</label>
                  <input
                    type="number"
                    placeholder="default"
                    value={advanced[k] ?? ""}
                    onChange={(e) => setAdvanced((prev) => {
                      const next = { ...prev };
                      if (e.target.value === "") delete next[k];
                      else next[k] = Number(e.target.value);
                      return next;
                    })}
                    className="w-full rounded border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs px-2 py-1.5 font-mono"
                  />
                </div>
              ))}
            </div>
          )}
        </div>

        {phase === "setup" && (
          <button
            onClick={start}
            disabled={starting}
            className="self-start px-8 py-2.5 rounded-lg bg-primary text-white font-bold hover:opacity-90 transition-opacity flex items-center gap-2 disabled:opacity-50"
          >
            <span className="material-symbols-outlined text-base">rocket_launch</span>
            {starting ? "…" : COPY.start}
          </button>
        )}
      </div>

      {/* Progress + chart */}
      {phase !== "setup" && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            {phase === "running" && (
              <>
                <span className="material-symbols-outlined text-primary animate-spin">sync</span>
                <span className="font-semibold text-slate-600 dark:text-slate-300">{COPY.running}</span>
              </>
            )}
            {phase === "done" && (
              <>
                <span className="material-symbols-outlined text-green-500">check_circle</span>
                <span className="font-semibold">{COPY.resultsTitle}</span>
              </>
            )}
            <span className="ml-auto text-sm font-mono text-slate-400">
              {COPY.testedOf(tested.length, run?.n_configs)}
            </span>
          </div>

          {phase === "done" && <p className="text-sm text-slate-500">{COPY.resultsBlurb}</p>}

          {run?.resolution_warning && (
            <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-600 dark:text-amber-400 flex items-start gap-2">
              <span className="material-symbols-outlined text-base shrink-0">warning</span>
              <span><span className="font-bold">{COPY.lowResolution}</span> {run.resolution_warning}</span>
            </div>
          )}

          {run?.status === "failed" ? (
            <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-500">
              {run.error ?? COPY.failed}
            </div>
          ) : tested.length === 0 && phase === "done" ? (
            <p className="text-sm text-slate-400 italic">{COPY.noResults}</p>
          ) : (
            <>
              <Plot
                data={traces}
                layout={{
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  xaxis: { title: { text: COPY.axisAttack }, ticksuffix: "%", gridcolor: "#e2e8f0", rangemode: "tozero" },
                  yaxis: { title: { text: COPY.axisQuality }, ticksuffix: "%", gridcolor: "#e2e8f0" },
                  shapes,
                  showlegend: false,
                  margin: { t: 30, r: 40, b: 60, l: 60 },
                  font: { family: "Inter, sans-serif", color: "#94a3b8" },
                  hovermode: "closest",
                  transition: { duration: 400, easing: "cubic-in-out" },
                }}
                config={{ responsive: true, displaylogo: false, modeBarButtons: [["toImage"]] }}
                style={{ width: "100%", height: 440 }}
                useResizeHandler
                onClick={(e) => {
                  // Only the highlighted trace is selectable; it is always drawn last.
                  const point = e.points?.[0];
                  if (point == null || point.curveNumber !== traces.length - 1) return;
                  const hit = highlights[point.pointNumber];
                  if (hit) { setSelected(hit); setVerification(null); setAdopted(false); }
                }}
              />
              <p className="text-xs text-slate-400">{COPY.fprNote}</p>
              {qualityFloor !== undefined && (
                <p className="text-xs text-slate-400">
                  <span className="inline-block w-6 border-t border-dashed border-slate-400 align-middle mr-1" />
                  {COPY.qualityLimitLine} — {pct(qualityFloor)}
                </p>
              )}
            </>
          )}
        </div>
      )}

      {selected && (
        <SettingPanel
          setting={selected}
          verification={verification}
          optimistic={!!optimistic}
          adopted={adopted}
          onClose={() => { setSelected(null); setVerification(null); setAdopted(false); }}
          onConfirm={() => confirm(selected)}
          onAdopt={adopt}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Side panel: the one place raw hyperparameters are appropriate.
// ---------------------------------------------------------------------------

function StatCard({ label, value, tone }: { label: string; value: string; tone?: "amber" }) {
  return (
    <div className={`rounded-lg px-4 py-3 border ${
      tone === "amber"
        ? "border-amber-500/40 bg-amber-500/10"
        : "border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900"
    }`}>
      <p className="text-xs text-slate-400">{label}</p>
      <p className="font-mono font-bold text-lg">{value}</p>
    </div>
  );
}

function SettingPanel({
  setting, verification, optimistic, adopted, onClose, onConfirm, onAdopt,
}: {
  setting: Setting;
  verification: Verification | null;
  optimistic: boolean;
  adopted: boolean;
  onClose: () => void;
  onConfirm: () => void;
  onAdopt: () => void;
}) {
  // The non-private anchor carries eps = inf, which JSON cannot represent: the
  // backend has to send null (or a string) for it, so treat anything
  // non-finite or absent as "no protection" rather than printing NaN.
  const epsilon = setting.epsilon;
  const epsilonLabel =
    epsilon == null ? "none" : Number.isFinite(epsilon) ? epsilon.toFixed(2) : "none";

  const rows: Array<[string, string]> = [
    ["Privacy budget (ε)", epsilonLabel],
    ["Noise multiplier", setting.config.noise_multiplier?.toFixed(3) ?? "—"],
    ["Clipping norm", setting.config.max_grad_norm?.toFixed(3) ?? "—"],
    ["Learning rate", setting.config.learning_rate?.toExponential(2) ?? "—"],
    ["Batch size", setting.config.batch_size?.toString() ?? "—"],
  ];

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <aside className="relative w-full max-w-md h-full overflow-y-auto bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 p-6 flex flex-col gap-6">
        <div className="flex items-start justify-between">
          <h3 className="text-xl font-black">{COPY.panelTitle}</h3>
          <button onClick={onClose} className="material-symbols-outlined text-slate-400 hover:text-slate-600">close</button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <StatCard label={COPY.estimatedAttack} value={`${(setting.attack_tpr! * 100).toFixed(2)}%`} />
          <StatCard label={COPY.estimatedQuality} value={`${(setting.utility * 100).toFixed(2)}%`} />
        </div>

        <div className="rounded-lg border border-slate-200 dark:border-slate-800 divide-y divide-slate-200 dark:divide-slate-800">
          {rows.map(([label, value]) => (
            <div key={label} className="flex items-center justify-between px-4 py-2.5 text-sm">
              <span className="text-slate-500">{label}</span>
              <span className="font-mono font-semibold">{value}</span>
            </div>
          ))}
        </div>

        {verification == null && (
          <button
            onClick={onConfirm}
            className="px-6 py-2.5 rounded-lg bg-primary text-white font-bold hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
          >
            <span className="material-symbols-outlined text-base">verified</span>
            {COPY.confirm}
          </button>
        )}

        {verification?.status === "running" && (
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 px-4 py-4 flex flex-col gap-2">
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-primary animate-spin">sync</span>
              <span className="font-semibold text-sm">{COPY.verifying}</span>
            </div>
            <p className="text-xs text-slate-400">{COPY.verifyingHelp}</p>
          </div>
        )}

        {verification?.status === "failed" && (
          <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-500">
            {verification.error ?? COPY.failed}
          </div>
        )}

        {verification?.verified && (
          <div className="flex flex-col gap-3">
            {optimistic && (
              <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs font-bold text-amber-600 dark:text-amber-400 flex items-center gap-2">
                <span className="material-symbols-outlined text-sm">warning</span>
                {COPY.optimistic}
              </div>
            )}
            <div className="grid grid-cols-2 gap-3">
              <StatCard label={`${COPY.estimate} — attack`} value={`${(verification.estimated.attack_tpr * 100).toFixed(2)}%`} />
              <StatCard
                label={`${COPY.verified} — attack`}
                value={`${(verification.verified.attack_tpr * 100).toFixed(2)}%`}
                tone={optimistic ? "amber" : undefined}
              />
              <StatCard label={`${COPY.estimate} — quality`} value={`${(verification.estimated.utility * 100).toFixed(2)}%`} />
              <StatCard label={`${COPY.verified} — quality`} value={`${(verification.verified.utility * 100).toFixed(2)}%`} />
            </div>

            {adopted ? (
              <p className="text-sm font-semibold text-green-600 dark:text-green-400 flex items-center gap-2">
                <span className="material-symbols-outlined text-base">check_circle</span>
                {COPY.adopted}
              </p>
            ) : (
              <button
                onClick={onAdopt}
                className="px-6 py-2.5 rounded-lg bg-primary text-white font-bold hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
              >
                <span className="material-symbols-outlined text-base">check</span>
                {COPY.adopt}
              </button>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}
