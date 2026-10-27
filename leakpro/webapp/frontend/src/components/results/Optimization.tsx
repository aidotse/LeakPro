import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { api, ModelResult, OptimizationRun, Setting, Verification } from "../../api";
import { COPY, fprLabel, tagsFor } from "./optimizationCopy";

const ACCENT = "#f5a623";
const TESTED = "#94a3b8";
const BASELINE = "#ef4444";
const POLL_MS = 2000;
const MAX_HIGHLIGHTS = 5;

interface Props {
  jobId: string;
  model: ModelResult;
  onBack: () => void;
  /** Called after the backend has stored the adopted row; the parent re-reads the job's results. */
  onAdopted: (setting: Setting, verification: Verification) => void;
}

const pct = (v: number | null | undefined, digits = 1) =>
  v == null ? "—" : `${(v * 100).toFixed(digits)}%`;

const rangeLabel = (ci: [number, number] | null | undefined) =>
  ci == null ? null : `${pct(ci[0], 2)} – ${pct(ci[1], 2)}`;

/** A setting whose likely range of attack success includes the false-alarm rate itself. */
function atChance(s: Pick<Setting, "attack_tpr_ci">, proxyFpr: number | null | undefined): boolean {
  if (s.attack_tpr_ci == null || proxyFpr == null) return false;
  return s.attack_tpr_ci[0] <= proxyFpr && proxyFpr <= s.attack_tpr_ci[1];
}

/**
 * The 3-5 settings the view labels, spread evenly over the frontier so the
 * tags always describe genuinely different choices. The frontier passed in is
 * already filtered by the quality floor and sorted safest-first.
 */
function pickHighlights(frontier: Setting[], max = MAX_HIGHLIGHTS): Setting[] {
  if (frontier.length <= max) return frontier;
  const step = (frontier.length - 1) / (max - 1);
  return Array.from({ length: max }, (_, i) => frontier[Math.round(i * step)]);
}

const isAuthError = (e: unknown) => e instanceof Error && e.message.startsWith("Unauthorized");

export default function Optimization({ jobId, model, onBack, onAdopted }: Props) {
  // A model loaded from a previous session belongs to its own job, which owns
  // the dataset and architecture the search retrains from. Optimize against
  // that job under the model's original name, not the current session's job or
  // the display name the compare view may have renamed it to.
  const targetJob = model.job_id ?? jobId;
  const targetName = model.orig_model_name ?? model.model_name;

  const [run, setRun] = useState<OptimizationRun | null>(null);
  const [maxQualityLoss, setMaxQualityLoss] = useState(0.05);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [advanced, setAdvanced] = useState<Record<string, number>>({});
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  // One verification per setting, so reselecting a measured setting shows its
  // result instead of resetting, and a failed one can be retried.
  const [verifications, setVerifications] = useState<Record<number, Verification>>({});
  const [adoptedIndex, setAdoptedIndex] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [authLost, setAuthLost] = useState(false);

  const fail = useCallback((e: unknown) => {
    if (isAuthError(e)) { setAuthLost(true); setError(COPY.authLost); return; }
    setError(e instanceof Error ? e.message : String(e));
  }, []);

  // ── polling ──────────────────────────────────────────────────────────────
  // A setTimeout chain, never setInterval: requests cannot overlap, so a late
  // "running" response can never land after "done" and flip the view back. The
  // sequence number drops responses from a chain that was replaced or unmounted.
  const pollSeq = useRef(0);
  const fetchRun = useCallback(async (seq: number) => {
    try {
      const next = await api.getOptimization(targetJob, targetName);
      if (seq !== pollSeq.current) return null;
      setRun(next);
      setError(null);
      return next;
    } catch (e) {
      if (seq !== pollSeq.current) return null;
      fail(e);
      return null;
    }
  }, [targetJob, targetName, fail]);

  // The chain (re)starts whenever the run enters "running", which is how a
  // start or a resume picks up polling without a second loop.
  const polling = run?.status === "running";
  useEffect(() => {
    const seq = ++pollSeq.current;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      const next = await fetchRun(seq);
      if (seq !== pollSeq.current) return;
      if (next?.status === "running" && !authLost) timer = setTimeout(tick, POLL_MS);
    };
    tick();
    return () => { pollSeq.current++; if (timer) clearTimeout(timer); };
  }, [fetchRun, authLost, polling]);

  const phase: "setup" | "running" | "done" =
    run == null || run.status === "idle" ? "setup" : run.status === "running" ? "running" : "done";

  const start = async () => {
    setStarting(true);
    setError(null);
    try {
      await api.startOptimization(targetJob, targetName, { max_quality_loss: maxQualityLoss, advanced });
      // One read: the status flips to "running" and the polling effect takes over.
      await fetchRun(++pollSeq.current);
    } catch (e) {
      fail(e);
    } finally {
      setStarting(false);
    }
  };

  // ── derived ──────────────────────────────────────────────────────────────
  // The comparison basis is an unprotected model trained on the same subset as
  // every trial, so the gap is the cost of protection alone. The user's own
  // model's accuracy is shown next to it but never used as the floor: its
  // training set is a different size.
  const baseline = run?.baseline_utility ?? undefined;
  const qualityFloor = baseline !== undefined ? baseline * (1 - maxQualityLoss) : undefined;
  const proxyFpr = run?.proxy_fpr ?? undefined;

  const tested = useMemo(
    () => (run?.settings ?? []).filter((s) => s.attack_tpr != null),
    [run?.settings],
  );
  const unmeasured = (run?.settings.length ?? 0) - tested.length;
  const withinLimit = useCallback(
    (s: Setting) => qualityFloor === undefined || s.utility >= qualityFloor,
    [qualityFloor],
  );

  // The frontier: every best setting, safest first. Filtered by the quality
  // floor BEFORE the highlights are thinned, so "Safest" is always a setting
  // the user can actually take, and every frontier point is selectable.
  const frontier = useMemo(() => {
    if (!run) return [];
    const byIndex = new Map(run.settings.map((s) => [s.index, s]));
    return run.best_indices
      .map((i) => byIndex.get(i))
      .filter((s): s is Setting => s != null && s.attack_tpr != null)
      .sort((a, b) => a.attack_tpr! - b.attack_tpr!);
  }, [run]);
  const frontierWithin = useMemo(() => frontier.filter(withinLimit), [frontier, withinLimit]);
  // Settings the audit cannot tell from guessing are shown but not ranked or tagged.
  const chanceCount = tested.filter((s) => atChance(s, proxyFpr)).length;
  const highlights = useMemo(
    () => pickHighlights(frontierWithin.filter((s) => !atChance(s, proxyFpr))),
    [frontierWithin, proxyFpr],
  );
  const tags = tagsFor(highlights.length);
  const tagOf = (s: Setting) => {
    const i = highlights.findIndex((h) => h.index === s.index);
    return i >= 0 ? tags[i] : atChance(s, proxyFpr) ? COPY.chanceTag : undefined;
  };

  const selected = useMemo(
    () => (selectedIndex == null ? null : run?.settings.find((s) => s.index === selectedIndex) ?? null),
    [run?.settings, selectedIndex],
  );
  const verification = selectedIndex == null ? null : verifications[selectedIndex] ?? null;

  // ── verification ─────────────────────────────────────────────────────────
  const confirm = async (setting: Setting) => {
    setError(null);
    try {
      const v = await api.verifySetting(targetJob, targetName, setting.index);
      setVerifications((prev) => ({ ...prev, [v.index]: v }));
    } catch (e) {
      fail(e);
    }
  };

  // Poll every running verification by its own index, so a result can never
  // land in another setting's panel. Errors stop that verification's polling
  // and are shown; a 404 means the backend lost the record.
  const running = Object.values(verifications).filter((v) => v.status === "running").map((v) => v.index);
  const runningKey = running.join(",");
  useEffect(() => {
    if (!runningKey || authLost) return;
    let cancelled = false;
    const timers: ReturnType<typeof setTimeout>[] = [];
    runningKey.split(",").map(Number).forEach((index) => {
      const tick = async () => {
        if (cancelled) return;
        try {
          const v = await api.getVerification(targetJob, targetName, index);
          if (cancelled) return;
          setVerifications((prev) => ({ ...prev, [index]: v }));
          if (v.status === "running") timers.push(setTimeout(tick, POLL_MS));
        } catch (e) {
          if (cancelled) return;
          fail(e);
          setVerifications((prev) => ({
            ...prev,
            [index]: { ...prev[index], status: "failed", error: e instanceof Error ? e.message : String(e) },
          }));
        }
      };
      timers.push(setTimeout(tick, POLL_MS));
    });
    return () => { cancelled = true; timers.forEach(clearTimeout); };
  }, [runningKey, targetJob, targetName, authLost, fail]);

  const adopt = async () => {
    if (!selected || !verification || verification.index !== selected.index) return;
    try {
      await api.adoptSetting(targetJob, targetName, selected.index);
      setAdoptedIndex(selected.index);
      onAdopted(selected, verification);
    } catch (e) {
      fail(e);
    }
  };

  // ── chart ────────────────────────────────────────────────────────────────
  const frontierIndices = new Set(frontier.map((s) => s.index));
  const backdrop = phase === "done" ? tested.filter((s) => !frontierIndices.has(s.index)) : tested;
  const hover = (s: Setting) => {
    const parts = [`Attack success ${pct(s.attack_tpr, 2)}`];
    const r = rangeLabel(s.attack_tpr_ci);
    if (r) parts.push(`${COPY.range} ${r}`);
    parts.push(`Quality ${pct(s.utility, 2)}`);
    const tag = tagOf(s);
    if (tag) parts.push(`<b>${tag}</b>`);
    if (!withinLimit(s)) parts.push(`<b>${COPY.exceedsLimit}</b>`);
    return parts.join("<br>");
  };

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
      text: backdrop.map(hover),
      hovertemplate: "%{text}<extra></extra>",
    },
  ];

  if (run?.baseline_attack_tpr != null && baseline !== undefined) {
    traces.push({
      x: [run.baseline_attack_tpr * 100],
      y: [baseline * 100],
      mode: "markers+text",
      type: "scatter",
      name: COPY.baselinePoint,
      text: [COPY.baselinePoint],
      textposition: "bottom right",
      textfont: { size: 11, color: BASELINE },
      marker: { size: 11, color: BASELINE, symbol: "diamond" },
      hovertemplate: `${COPY.baselinePoint}: attack success %{x:.2f}%, quality %{y:.2f}%<extra></extra>`,
    });
  }

  if (phase === "done" && frontier.length > 0) {
    traces.push({
      x: frontier.map((s) => s.attack_tpr! * 100),
      y: frontier.map((s) => s.utility * 100),
      mode: "lines",
      type: "scatter",
      line: { color: ACCENT, width: 2 },
      hoverinfo: "skip",
      showlegend: false,
    });
    // Every frontier point is drawn in this trace and is clickable; the tagged
    // ones are bigger. Chance-level settings are hollow.
    traces.push({
      x: frontier.map((s) => s.attack_tpr! * 100),
      y: frontier.map((s) => s.utility * 100),
      text: frontier.map((s) => (highlights.some((h) => h.index === s.index) ? tagOf(s) ?? "" : "")),
      mode: "text+markers",
      type: "scatter",
      name: COPY.resultsTitle,
      textposition: "top right",
      textfont: { size: 12, color: ACCENT },
      marker: {
        size: frontier.map((s) => (highlights.some((h) => h.index === s.index) ? 15 : 10)) as unknown as number,
        color: frontier.map((s) => (atChance(s, proxyFpr) ? "#ffffff" : ACCENT)) as unknown as string,
        // plotly.js accepts per-point arrays here; its typings do not.
        opacity: frontier.map((s) => (withinLimit(s) ? 1 : 0.3)) as unknown as number,
        line: { width: 2, color: ACCENT },
      },
      customdata: frontier.map(hover),
      hovertemplate: "%{customdata}<extra></extra>",
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

  const select = (s: Setting) => setSelectedIndex(s.index);

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
          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-300 dark:border-surface-border text-sm font-bold hover:bg-slate-100 dark:hover:bg-surface-2 transition-colors shrink-0"
        >
          <span className="material-symbols-outlined text-base">arrow_back</span>
          {COPY.back}
        </button>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-500">
          {error}
        </div>
      )}

      {/* Constraint — one control, everything else behind Advanced */}
      <div className="rounded-xl border border-slate-200 dark:border-surface-border p-5 flex flex-col gap-4">
        <div className="flex items-center gap-6 flex-wrap">
          <div className="flex-1 min-w-[280px]">
            <label htmlFor="max-quality-loss" className="text-sm font-bold block mb-2">
              {COPY.constraintLabel}:{" "}
              <span className="text-primary font-mono">{(maxQualityLoss * 100).toFixed(0)}%</span>
            </label>
            <input
              id="max-quality-loss"
              type="range" min={0} max={0.5} step={0.01}
              value={maxQualityLoss}
              onChange={(e) => setMaxQualityLoss(Number(e.target.value))}
              className="w-full accent-primary"
              aria-valuetext={`${(maxQualityLoss * 100).toFixed(0)}%`}
            />
            <p className="text-xs text-slate-400 mt-1">{COPY.constraintHelp}</p>
          </div>
          <div className="flex gap-6 text-sm">
            <div>
              <p className="text-xs text-slate-400">{COPY.baselineLabel}</p>
              <p className="font-mono font-bold text-lg">{pct(baseline)}</p>
              {run?.n_members != null && baseline !== undefined && (
                <p className="text-[11px] text-slate-400 max-w-[180px]">{COPY.baselineHelp(run.n_members)}</p>
              )}
            </div>
            <div>
              <p className="text-xs text-slate-400">{COPY.floorLabel}</p>
              <p className="font-mono font-bold text-lg">{pct(qualityFloor)}</p>
            </div>
            {(run?.model_utility ?? model.test_accuracy) != null && (
              <div>
                <p className="text-xs text-slate-400">{COPY.modelUtilityLabel}</p>
                <p className="font-mono font-bold text-lg">{pct(run?.model_utility ?? model.test_accuracy)}</p>
              </div>
            )}
          </div>
        </div>
        {run?.n_members != null && <p className="text-xs text-slate-400">{COPY.scaleNote(run.n_members)}</p>}

        <div>
          <button
            onClick={() => setAdvancedOpen(!advancedOpen)}
            aria-expanded={advancedOpen}
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
                  <label htmlFor={`advanced-${k}`} className="text-xs font-semibold text-slate-500 mb-1 block">{k}</label>
                  <input
                    id={`advanced-${k}`}
                    type="number"
                    placeholder="searched"
                    value={advanced[k] ?? ""}
                    onChange={(e) => setAdvanced((prev) => {
                      const next = { ...prev };
                      if (e.target.value === "") delete next[k];
                      else next[k] = Number(e.target.value);
                      return next;
                    })}
                    className="w-full rounded border-slate-300 dark:border-surface-border bg-white dark:bg-surface text-xs px-2 py-1.5 font-mono"
                  />
                </div>
              ))}
            </div>
          )}
        </div>

        {/* A failed run must stay startable: hiding the button here once locked
            the view permanently on a stale error read back from disk. */}
        {(phase === "setup" || run?.status === "failed") && (
          <button
            onClick={start}
            disabled={starting || authLost}
            className="self-start px-8 py-2.5 rounded-lg bg-slate-700 text-cream border border-primary font-bold hover:bg-slate-600 transition-colors flex items-center gap-2 disabled:opacity-50"
          >
            <span className="material-symbols-outlined text-base">{run?.status === "failed" ? "refresh" : "rocket_launch"}</span>
            {starting ? "…" : run?.status === "failed" ? COPY.retry : COPY.start}
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
              {COPY.testedOf(run?.settings.length ?? 0, run?.n_trials ?? undefined)}
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
            <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-500 flex flex-col gap-1">
              <span className="font-bold">{COPY.failed} {COPY.retryHint}</span>
              <span className="text-xs opacity-80 break-words">{run.error ?? ""}</span>
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
                  // The frontier trace is always drawn last; every point on it selects.
                  const point = e.points?.[0];
                  if (point == null || point.curveNumber !== traces.length - 1) return;
                  const hit = frontier[point.pointNumber];
                  if (hit) select(hit);
                }}
              />
              <p className="text-xs text-slate-400">{COPY.fprNote(proxyFpr)}</p>
              {chanceCount > 0 && (
                <p className="text-xs text-slate-400">{COPY.chanceNote(chanceCount, proxyFpr)}</p>
              )}
              {unmeasured > 0 && (
                <p className="text-xs text-slate-400">{COPY.unmeasured(unmeasured)}</p>
              )}
              {qualityFloor !== undefined && (
                <p className="text-xs text-slate-400">
                  <span className="inline-block w-6 border-t border-dashed border-slate-400 align-middle mr-1" />
                  {COPY.qualityLimitLine} — {pct(qualityFloor)}
                </p>
              )}

              {/* Keyboard-reachable list of every frontier setting; the chart is the picture, not the only way in. */}
              {phase === "done" && frontier.length > 0 && (
                <div className="flex flex-col gap-1 mt-2">
                  <p className="text-xs font-bold uppercase tracking-wider text-slate-400">{COPY.frontierList}</p>
                  {frontierWithin.length === 0 && <p className="text-xs text-slate-400 italic">{COPY.frontierNone}</p>}
                  <ul className="flex flex-wrap gap-2">
                    {frontier.map((s) => {
                      const tag = tagOf(s);
                      const within = withinLimit(s);
                      return (
                        <li key={s.index}>
                          <button
                            onClick={() => select(s)}
                            aria-pressed={selectedIndex === s.index}
                            title={within ? undefined : COPY.exceedsLimit}
                            className={`px-3 py-1.5 rounded-lg border text-xs font-mono transition-colors ${
                              selectedIndex === s.index
                                ? "border-primary bg-primary/10 text-primary"
                                : "border-slate-200 dark:border-surface-border hover:border-primary/60"
                            } ${within ? "" : "opacity-50"}`}
                          >
                            {tag && <span className="font-sans font-bold mr-2">{tag}</span>}
                            {pct(s.attack_tpr, 2)} / {pct(s.utility, 1)}
                            {verifications[s.index]?.status === "done" && (
                              <span className="material-symbols-outlined text-sm align-middle ml-1 text-green-500">verified</span>
                            )}
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {selected && (
        <SettingPanel
          setting={selected}
          run={run}
          verification={verification?.index === selected.index ? verification : null}
          adopted={adoptedIndex === selected.index}
          onClose={() => setSelectedIndex(null)}
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

function StatCard({ label, value, sub, tone }: { label: string; value: string; sub?: string | null; tone?: "amber" }) {
  return (
    <div className={`rounded-lg px-4 py-3 border ${
      tone === "amber"
        ? "border-amber-500/40 bg-amber-500/10"
        : "border-slate-200 dark:border-surface-border bg-slate-50 dark:bg-surface"
    }`}>
      <p className="text-xs text-slate-400">{label}</p>
      <p className="font-mono font-bold text-lg">{value}</p>
      {sub && <p className="text-[11px] text-slate-400 font-mono">{sub}</p>}
    </div>
  );
}

/** ε as the user should read it: a number for one run, or "no guarantee"; never "none", which reads as zero. */
function epsilonLabel(epsilon: number | null | undefined): string {
  return epsilon == null || !Number.isFinite(epsilon) ? COPY.epsilonNone : epsilon.toFixed(2);
}

function SettingPanel({
  setting, run, verification, adopted, onClose, onConfirm, onAdopt,
}: {
  setting: Setting;
  run: OptimizationRun | null;
  verification: Verification | null;
  adopted: boolean;
  onClose: () => void;
  onConfirm: () => void;
  onAdopt: () => void;
}) {
  const verified = verification?.verified ?? null;
  const proxyFpr = run?.proxy_fpr ?? undefined;
  const estimatedCi = verification?.estimated.attack_tpr_ci ?? setting.attack_tpr_ci ?? null;
  const estimatedTpr = verification?.estimated.attack_tpr ?? setting.attack_tpr ?? null;

  // "Optimistic" is decided on intervals, not on a relative margin: a 20%
  // difference between two numbers each resting on a handful of records is
  // noise. The measured interval must lie entirely above the estimated one.
  // An estimate of exactly 0 is covered the same way (its interval is [0, hi]).
  const optimistic =
    verified?.attack_tpr != null && verified.attack_tpr_ci != null && estimatedCi != null &&
    verified.attack_tpr_ci[0] > estimatedCi[1];
  const measuredAtChance = verified != null && atChance(verified, verified.proxy_fpr ?? proxyFpr);

  const rows: Array<[string, string]> = [
    [COPY.epsilonLabel, epsilonLabel(verified ? verified.epsilon : setting.epsilon)],
    ["Noise multiplier", setting.config.noise_multiplier?.toFixed(3) ?? "—"],
    ["Clipping norm", setting.config.max_grad_norm?.toFixed(3) ?? "—"],
    ["Learning rate", setting.config.learning_rate?.toExponential(2) ?? "—"],
    ["Batch size", setting.config.batch_size?.toString() ?? "—"],
  ];

  const qualityLoss =
    verified?.utility != null && verified.baseline_utility != null && verified.baseline_utility > 0
      ? 1 - verified.utility / verified.baseline_utility
      : null;
  const measuredFpr = verified?.realized_fpr ?? verified?.proxy_fpr ?? proxyFpr;
  const canConfirm = verification == null || verification.status === "failed";

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <aside
        role="dialog"
        aria-label={COPY.panelTitle}
        className="relative w-full max-w-md h-full overflow-y-auto bg-white dark:bg-surface border-l border-slate-200 dark:border-surface-border p-6 flex flex-col gap-6"
      >
        <div className="flex items-start justify-between">
          <h3 className="text-xl font-black">{COPY.panelTitle}</h3>
          <button onClick={onClose} aria-label="Close" className="material-symbols-outlined text-slate-400 hover:text-slate-600">close</button>
        </div>

        <div className="flex flex-col gap-2">
          <div className="grid grid-cols-2 gap-3">
            <StatCard label={COPY.estimatedAttack} value={pct(setting.attack_tpr, 2)} sub={rangeLabel(setting.attack_tpr_ci)} />
            <StatCard label={COPY.estimatedQuality} value={pct(setting.utility, 2)} />
          </div>
          {(setting.n_members ?? run?.n_members) != null && (
            <p className="text-xs text-slate-400">{COPY.estimatedScale((setting.n_members ?? run?.n_members)!)}</p>
          )}
          {atChance(setting, proxyFpr) && (
            <p className="text-xs text-slate-400">{COPY.chanceNote(1, proxyFpr)}</p>
          )}
        </div>

        <div className="rounded-lg border border-slate-200 dark:border-surface-border divide-y divide-slate-200 dark:divide-surface-border">
          {rows.map(([label, value]) => (
            <div key={label} className="flex items-center justify-between px-4 py-2.5 text-sm">
              <span className="text-slate-500">{label}</span>
              <span className="font-mono font-semibold">{value}</span>
            </div>
          ))}
          <p className="px-4 py-2 text-[11px] text-slate-400">{COPY.epsilonNote}</p>
        </div>

        {canConfirm && (
          <div className="flex flex-col gap-2">
            <button
              onClick={onConfirm}
              className="px-6 py-2.5 rounded-lg bg-slate-700 text-cream border border-primary font-bold hover:bg-slate-600 transition-colors flex items-center justify-center gap-2"
            >
              <span className="material-symbols-outlined text-base">verified</span>
              {verification?.status === "failed" ? COPY.retryConfirm : COPY.confirm}
            </button>
            <p className="text-xs text-slate-400">{COPY.confirmHelp}</p>
          </div>
        )}

        {verification?.status === "running" && (
          <div className="rounded-xl border border-slate-200 dark:border-surface-border px-4 py-4 flex flex-col gap-2" aria-live="polite">
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-primary animate-spin">sync</span>
              <span className="font-semibold text-sm">{COPY.verifying}</span>
            </div>
            <p className="text-xs text-slate-400">{COPY.verifyingHelp}</p>
          </div>
        )}

        {verification?.status === "failed" && (
          <div role="alert" className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-500">
            {verification.error ?? COPY.failed}
          </div>
        )}

        {verified && (
          <div className="flex flex-col gap-3">
            <div className={`rounded-lg px-3 py-2 text-xs flex items-start gap-2 border ${
              verified.scope === "user_model"
                ? "border-green-500/40 bg-green-500/10 text-green-700 dark:text-green-400"
                : "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400"
            }`}>
              <span className="material-symbols-outlined text-sm shrink-0">{verified.scope === "user_model" ? "person_check" : "science"}</span>
              <span>
                <span className="font-bold">{verified.scope === "user_model" ? COPY.verifiedOnYourModel : COPY.verifiedOnSubset}</span>
                {verified.scope !== "user_model" && <> {COPY.subsetScopeHelp}</>}
                {verified.n_members != null && verified.scope === "user_model" && (
                  <> {verified.n_members.toLocaleString()} training records, {verified.n_shadow} reference models,
                    false-alarm rate {measuredFpr != null ? fprLabel(measuredFpr) : "—"}
                    {verified.n_nonmember_false_positives != null && ` (${verified.n_nonmember_false_positives} false alarms behind the threshold)`}.</>
                )}
              </span>
            </div>
            {verified.notes.map((n) => (
              <p key={n} className="text-xs text-slate-400">{n}</p>
            ))}
            {optimistic && (
              <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs font-bold text-amber-600 dark:text-amber-400 flex items-center gap-2">
                <span className="material-symbols-outlined text-sm">warning</span>
                {COPY.optimistic}
              </div>
            )}
            {/* A low attack number can mean "well protected" or "we could not
                measure it". Those must never look the same to the reader. */}
            {(verified.resolution_warning || verified.attack_tpr == null) && (
              <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-600 dark:text-amber-400 flex items-start gap-2">
                <span className="material-symbols-outlined text-sm shrink-0">help</span>
                <span><span className="font-bold">{COPY.notMeasurable}</span> {verified.resolution_warning ?? COPY.notMeasurableHelp}</span>
              </div>
            )}
            {measuredAtChance && (
              <p className="text-xs text-slate-400">{COPY.chanceNote(1, verified.proxy_fpr ?? proxyFpr)}</p>
            )}
            <div className="grid grid-cols-2 gap-3">
              <StatCard label={`${COPY.estimate} — attack`} value={pct(estimatedTpr, 2)} sub={rangeLabel(estimatedCi)} />
              <StatCard
                label={`${COPY.verified} — attack`}
                value={pct(verified.attack_tpr, 2)}
                sub={rangeLabel(verified.attack_tpr_ci)}
                tone={optimistic ? "amber" : undefined}
              />
              <StatCard label={`${COPY.estimate} — quality`} value={pct(verification?.estimated.utility, 2)} />
              <StatCard
                label={`${COPY.verified} — quality`}
                value={pct(verified.utility, 2)}
                sub={qualityLoss != null ? `${COPY.qualityLoss} ${pct(qualityLoss, 1)} ${COPY.qualityLossHelp(pct(verified.baseline_utility, 1))}` : null}
              />
            </div>

            {adopted ? (
              <p className="text-sm font-semibold text-green-600 dark:text-green-400 flex items-center gap-2">
                <span className="material-symbols-outlined text-base">check_circle</span>
                {COPY.adopted}
              </p>
            ) : (
              <div className="flex flex-col gap-2">
                <button
                  onClick={onAdopt}
                  disabled={verified.attack_tpr == null}
                  className="px-6 py-2.5 rounded-lg bg-slate-700 text-cream border border-primary font-bold hover:bg-slate-600 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  <span className="material-symbols-outlined text-base">check</span>
                  {COPY.adopt}
                </button>
                <p className="text-xs text-slate-400">{COPY.adoptHelp}</p>
              </div>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}
