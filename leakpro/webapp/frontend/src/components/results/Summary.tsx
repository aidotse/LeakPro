import React, { useState } from "react";
import { ModelResult } from "../../api";
import MetaPanel from "./MetaPanel";

interface Props { results: ModelResult[] }

// ---------------------------------------------------------------------------
// Harm assessment = Vulnerability (measured by LeakPro) × Loss Magnitude (the
// user's use-case). Loss-Magnitude decomposition follows Sion et al. (data
// subject-aware privacy risk); computed in the spirit of a FAIR Bayesian network.
// The combine here is a transparent placeholder for that backend model.
// ---------------------------------------------------------------------------
const FPR = 0.001; // 0.1% — matches the TPR@0.1%FPR metric

const CITATION =
  "Sion, Van Landuyt, Wuyts & Joosen, “Privacy Risk Assessment for Data " +
  "Subject-aware Threat Modeling,” IEEE EuroS&PW, 2019 — Loss-Magnitude " +
  "decomposition, computed as a FAIR Bayesian network (Wang, Neil & Fenton, 2020).";

function bestAuc(model: ModelResult): number | undefined {
  const vals = model.attacks.map((a) => a.roc_auc).filter((v): v is number => v !== null && v !== undefined);
  return vals.length ? Math.max(...vals) : undefined;
}
function bestTpr(model: ModelResult): { value: number; attack: string } | undefined {
  let best: { value: number; attack: string } | undefined;
  model.attacks.forEach((a) => {
    const v = a.tpr_at_fpr?.["TPR@0.1%FPR"];
    if (v != null && (best === undefined || v > best.value)) best = { value: v, attack: a.attack_name };
  });
  return best;
}
function vulnMultiplier(model: ModelResult): number | undefined {
  const t = bestTpr(model);
  if (t && t.value > 0) return t.value / FPR;
  const auc = bestAuc(model);
  return auc !== undefined ? Math.max(1, (auc - 0.5) * 40) : undefined;
}
function pct(v: number | undefined) { return v !== undefined ? (v * 100).toFixed(1) + "%" : "—"; }
function defaultSubjects(m: ModelResult): number {
  const meta = (m as unknown as { train_meta?: { n_samples?: number }; n_samples?: number });
  return meta.train_meta?.n_samples ?? meta.n_samples ?? 10000;
}

const SENSITIVITY = [
  { key: "low", label: "Low", weight: 1 },
  { key: "moderate", label: "Moderate", weight: 2 },
  { key: "high", label: "High", weight: 3 },
  { key: "special", label: "Special-category", weight: 4 },
];
const SUBJECT_TYPES = [
  { key: "general", label: "General population", weight: 1 },
  { key: "vulnerable", label: "Vulnerable (minors, patients)", weight: 2 },
];

function InfoDot({ text }: { text: string }) {
  return (
    <span className="relative group inline-flex align-middle ml-1">
      <span className="material-symbols-outlined text-slate-400 hover:text-primary cursor-help text-sm leading-none">help</span>
      <span className="absolute left-1/2 -translate-x-1/2 top-5 z-30 hidden group-hover:block w-56 rounded-lg border border-slate-300 dark:border-surface-border bg-white dark:bg-surface-deep p-2.5 text-[11px] font-normal leading-snug text-slate-600 dark:text-slate-300 shadow-xl">
        {text}
      </span>
    </span>
  );
}

interface HarmInputs { sensitivity: string; subjectType: string; subjects: number; records: number }
type Verdict = { label: string; score: number; expSubjects: number; L: number; S: number; sensW: number; subjW: number; p: number; text: string };

// Placeholder combine — FAIR Risk = Vulnerability × Loss Magnitude.
// Returns a 0–100 risk index (real model would return a distribution) plus the
// concrete "expected re-identified subjects" = attack-success-prob × #subjects.
function harmVerdict(mult: number, tpr: number | undefined, h: HarmInputs): Verdict {
  const sensW = SENSITIVITY.find((s) => s.key === h.sensitivity)?.weight ?? 3;
  const subjW = SUBJECT_TYPES.find((s) => s.key === h.subjectType)?.weight ?? 1;
  const L = mult >= 10 ? 3 : mult >= 3 ? 2 : 1;                            // likelihood band (1–3)
  const S = Math.min(8, (sensW + (h.subjects >= 10000 ? 1 : 0)) * subjW);  // loss magnitude band (1–8)
  const score = Math.round((L * S) / 24 * 100);                            // → 0–100 index
  const p = tpr ?? Math.min(1, mult * FPR);                                // attack success probability
  const expSubjects = Math.round(p * h.subjects * Math.max(1, h.records));
  const text = score >= 50 ? "text-red-400" : score >= 21 ? "text-orange-300" : "text-emerald-300";
  const label = score >= 50 ? "HIGH" : score >= 21 ? "MEDIUM" : "LOW";
  return { label, score, expSubjects, L, S, sensW, subjW, p, text };
}

export default function Summary({ results }: Props) {
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [wizard, setWizard] = useState<null | "intro" | "inputs">(null);
  const [explain, setExplain] = useState<ModelResult | null>(null);
  const [harm, setHarm] = useState<HarmInputs | null>(null);
  const [draft, setDraft] = useState<HarmInputs>({
    sensitivity: "high", subjectType: "vulnerable",
    subjects: results[0] ? defaultSubjects(results[0]) : 10000, records: 1,
  });

  const toggleInfo = (key: string) => setOpenKey((prev) => (prev === key ? null : key));
  const sensLabel = (k: string) => SENSITIVITY.find((s) => s.key === k)?.label ?? k;
  const subjLabel = (k: string) => SUBJECT_TYPES.find((s) => s.key === k)?.label ?? k;

  const downloadCSV = () => {
    const rows = ["model,model_class,tpr_at_0.1pct_fpr,vulnerability_x_random,roc_auc,train_accuracy,test_accuracy,dpsgd,risk_index"];
    results.forEach((m) => {
      const tpr = bestTpr(m); const mult = vulnMultiplier(m);
      const v = harm && mult !== undefined ? harmVerdict(mult, tpr?.value, harm) : null;
      rows.push([
        m.model_name, m.model_class ?? "", tpr?.value.toFixed(6) ?? "", mult?.toFixed(1) ?? "",
        bestAuc(m)?.toFixed(6) ?? "", m.train_accuracy?.toFixed(6) ?? "", m.test_accuracy?.toFixed(6) ?? "",
        m.dpsgd ? `eps=${m.target_epsilon}` : "No",
        v ? String(v.score) : "",
      ].join(","));
    });
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "leakpro_summary.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-8">
      {/* Vulnerability cards */}
      {results.length > 1 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {results.map((m) => {
            const mult = vulnMultiplier(m);
            return (
              <div key={`${m.job_id}/${m.model_name}`} className="rounded-xl p-4 bg-slate-50 dark:bg-surface border border-slate-200 dark:border-surface-border">
                <p className="text-2xl font-black text-primary">{mult ? `${mult.toFixed(0)}×` : "—"}</p>
                <p className="text-[11px] uppercase tracking-wider text-slate-400 -mt-0.5">vs random</p>
                <p className="font-bold text-sm mt-2">{m.model_name}</p>
                {m.model_class && <p className="text-xs font-mono text-slate-400 mt-0.5">{m.model_class}</p>}
                {m.dpsgd && <p className="text-xs text-primary/80 mt-1">DP-SGD{m.target_epsilon != null ? ` ε=${m.target_epsilon}` : ""}</p>}
              </div>
            );
          })}
        </div>
      )}

      {/* Harm assessment bar */}
      <div className="flex items-center justify-between gap-3 flex-wrap rounded-xl border border-slate-200 dark:border-surface-border bg-slate-50 dark:bg-surface px-4 py-3">
        <div>
          <p className="font-bold text-sm">Harm assessment</p>
          <p className="text-xs text-slate-400">
            {harm ? "Risk tailored to your use case (see the Risk column)." : "Turn measured vulnerability into use-case risk."}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {harm && (
            <div className="relative group">
              <span className="material-symbols-outlined text-slate-400 hover:text-primary cursor-help text-xl">help</span>
              <div className="absolute right-0 top-7 z-20 hidden group-hover:block w-64 rounded-lg border border-slate-300 dark:border-surface-border bg-white dark:bg-surface-deep p-3 text-xs shadow-xl">
                <p className="font-bold mb-1.5 text-slate-500">Your inputs</p>
                <ul className="space-y-1 text-slate-600 dark:text-slate-300">
                  <li className="flex justify-between gap-3"><span className="text-slate-400">Sensitivity</span><span className="font-semibold">{sensLabel(harm.sensitivity)}</span></li>
                  <li className="flex justify-between gap-3"><span className="text-slate-400">Subject type</span><span className="font-semibold">{subjLabel(harm.subjectType)}</span></li>
                  <li className="flex justify-between gap-3"><span className="text-slate-400">Subjects</span><span className="font-semibold">{harm.subjects.toLocaleString()}</span></li>
                  <li className="flex justify-between gap-3"><span className="text-slate-400">Records / subject</span><span className="font-semibold">{harm.records}</span></li>
                </ul>
              </div>
            </div>
          )}
          <button
            onClick={() => setWizard("intro")}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-slate-700 text-cream border border-primary text-sm font-bold hover:bg-slate-600 transition-colors"
          >
            <span className="material-symbols-outlined text-base">{harm ? "refresh" : "balance"}</span>
            {harm ? "Redo harm assessment" : "Assess the harm in your case"}
          </button>
        </div>
      </div>

      {/* Comparison table */}
      <div className="rounded-xl border border-slate-200 dark:border-surface-border overflow-hidden">
        <div className="flex items-center justify-between px-3 py-2 bg-slate-50 dark:bg-surface border-b border-slate-200 dark:border-surface-border">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Summary</span>
          <button onClick={downloadCSV} className="flex items-center gap-1 px-2 py-1 rounded border border-slate-300 dark:border-surface-border text-slate-500 text-xs font-bold hover:bg-slate-100 dark:hover:bg-surface-2 transition-colors">
            <span className="material-symbols-outlined text-sm">download</span>CSV
          </button>
        </div>
        <table className="w-full text-sm">
          <thead className="bg-slate-50 dark:bg-surface border-b border-slate-200 dark:border-surface-border">
            <tr>
              {["Model", "TPR@0.1%FPR", "Test Acc", "DP-SGD", "Vulnerability", "Risk (your case)"].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-surface-border">
            {results.map((m) => {
              const key = `${m.job_id}/${m.model_name}`;
              const tpr = bestTpr(m); const mult = vulnMultiplier(m);
              const verdict = harm && mult !== undefined ? harmVerdict(mult, tpr?.value, harm) : null;
              const isOpen = openKey === key;
              return (
                <React.Fragment key={key}>
                  <tr className="hover:bg-slate-50/50 dark:hover:bg-surface/50 transition-colors">
                    <td className="px-4 py-3 font-semibold">
                      <div className="flex items-center gap-2 flex-wrap">
                        {m.model_name}
                        {m.model_class && <span className="text-xs px-1.5 py-0.5 rounded bg-slate-100 dark:bg-surface-2 text-slate-500 font-mono">{m.model_class}</span>}
                        <button onClick={() => toggleInfo(key)} title="Show model info"
                          className={`material-symbols-outlined text-base transition-colors ${isOpen ? "text-primary" : "text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"}`}>info</button>
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono">
                      {tpr !== undefined ? <>{(tpr.value * 100).toFixed(2)}% <span className="text-slate-400 font-sans text-xs">({tpr.attack})</span></> : "—"}
                    </td>
                    <td className="px-4 py-3 font-mono">{pct(m.test_accuracy)}</td>
                    <td className="px-4 py-3">{m.dpsgd ? <span className="text-primary font-semibold">{m.target_epsilon != null ? `ε=${m.target_epsilon}` : "Yes"}</span> : <span className="text-slate-400">No</span>}</td>
                    <td className="px-4 py-3 font-mono font-bold text-primary">{mult ? `${mult.toFixed(0)}× random` : "—"}</td>
                    <td className="px-4 py-3">
                      {verdict
                        ? (
                          <div className="flex items-center gap-1.5">
                            <span className={`text-lg font-black ${verdict.text}`}>{verdict.score}</span>
                            <span className="text-slate-400 text-[10px]">/100</span>
                            <button onClick={() => setExplain(m)} title="How is this computed?"
                              className="material-symbols-outlined text-base text-slate-400 hover:text-primary transition-colors">calculate</button>
                          </div>
                        )
                        : <span className="text-slate-400 text-xs">not assessed</span>}
                    </td>
                  </tr>
                  {isOpen && (
                    <tr className="bg-slate-50 dark:bg-surface/60"><td colSpan={6} className="px-6 py-4"><MetaPanel r={m} /></td></tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ---- Wizard: step 1 intro (cite + decomposition) ---- */}
      {wizard === "intro" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => setWizard(null)}>
          <div className="bg-white dark:bg-surface rounded-2xl shadow-2xl w-full max-w-2xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-surface-border">
              <h3 className="font-bold text-lg">Harm assessment framework</h3>
              <button onClick={() => setWizard(null)} className="text-slate-400 hover:text-slate-200"><span className="material-symbols-outlined">close</span></button>
            </div>
            <div className="p-6 space-y-5">
              <p className="text-sm text-slate-600 dark:text-slate-300">
                Risk is assessed as <b>Vulnerability × Loss Magnitude</b>. LeakPro <b>measures the
                vulnerability</b> from the attack; you provide the use-case factors that determine
                the <b>loss magnitude</b>.
              </p>

              {/* decomposition diagram */}
              <div className="rounded-xl border border-slate-200 dark:border-surface-border p-3 text-slate-700 dark:text-slate-200">
                <svg viewBox="0 0 520 224" className="w-full h-auto" role="img" aria-label="Risk decomposition">
                  {(() => {
                    const slate = "rgba(148,163,184,0.5)";
                    const boxFill = "rgba(148,163,184,0.10)";
                    const amber = "#f5a623";
                    return (
                      <>
                        {/* connectors */}
                        <path d="M260 44 V58 M110 58 H410 M110 58 V92 M410 58 V92 M410 136 V184 M71 184 H449 M71 184 V190 M197 184 V190 M323 184 V190 M449 184 V190" fill="none" stroke={slate} strokeWidth="1.5" />
                        {/* RISK */}
                        <rect x="210" y="10" width="100" height="34" rx="9" fill="rgba(245,166,35,0.15)" stroke={amber} strokeWidth="1.5" />
                        <text x="260" y="32" textAnchor="middle" fontSize="14" fontWeight="800" fill={amber}>RISK</text>
                        {/* branch labels — above the row-2 cells, both amber */}
                        <text x="110" y="78" textAnchor="middle" fontSize="11" fontWeight="700" fill={amber}>measured by LeakPro</text>
                        <text x="410" y="78" textAnchor="middle" fontSize="11" fontWeight="700" fill={amber}>use-case dependent</text>
                        {/* × */}
                        <text x="260" y="120" textAnchor="middle" fontSize="18" fill={slate}>×</text>
                        {/* Vulnerability */}
                        <rect x="28" y="92" width="164" height="44" rx="9" fill={boxFill} stroke={slate} strokeWidth="1.5" />
                        <text x="110" y="114" textAnchor="middle" fontSize="13" fontWeight="700" fill="currentColor">Vulnerability</text>
                        <text x="110" y="130" textAnchor="middle" fontSize="10" fill="currentColor" opacity="0.6">P(attack succeeds)</text>
                        {/* Loss Magnitude */}
                        <rect x="328" y="92" width="164" height="44" rx="9" fill={boxFill} stroke={slate} strokeWidth="1.5" />
                        <text x="410" y="114" textAnchor="middle" fontSize="13" fontWeight="700" fill="currentColor">Loss Magnitude</text>
                        <text x="410" y="130" textAnchor="middle" fontSize="10" fill="currentColor" opacity="0.6">how bad if it leaks</text>
                        {/* LM factors — inline row */}
                        {["Data sensitivity", "Subject type", "# Subjects", "# Records"].map((f, i) => {
                          const x = 12 + i * 126;
                          return (
                            <g key={f}>
                              <rect x={x} y="190" width="118" height="24" rx="6" fill={boxFill} stroke={slate} strokeWidth="1" />
                              <text x={x + 59} y="205" textAnchor="middle" fontSize="10" fill="currentColor">{f}</text>
                            </g>
                          );
                        })}
                      </>
                    );
                  })()}
                </svg>
              </div>

              <p className="text-xs text-slate-400 leading-relaxed border-l-2 border-primary/40 pl-3">{CITATION}</p>
            </div>
            <div className="flex justify-end gap-3 px-6 py-4 border-t border-slate-200 dark:border-surface-border">
              <button onClick={() => setWizard(null)} className="px-5 py-2 rounded-lg border border-slate-300 dark:border-surface-border text-sm font-bold hover:bg-slate-100 dark:hover:bg-surface-2 transition-colors">Cancel</button>
              <button onClick={() => setWizard("inputs")} className="px-5 py-2 rounded-lg bg-slate-700 text-cream border border-primary text-sm font-bold hover:bg-slate-600 transition-colors">Accept &amp; continue →</button>
            </div>
          </div>
        </div>
      )}

      {/* ---- Explain the risk number ---- */}
      {explain && harm && (() => {
        const mult = vulnMultiplier(explain);
        const v = mult !== undefined ? harmVerdict(mult, bestTpr(explain)?.value, harm) : null;
        if (!v || mult === undefined) return null;
        const Row = ({ label, children }: { label: string; children: React.ReactNode }) => (
          <div className="flex items-start gap-3 py-2.5">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400 w-28 shrink-0 pt-0.5">{label}</span>
            <div className="text-sm flex-1">{children}</div>
          </div>
        );
        return (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => setExplain(null)}>
            <div className="bg-white dark:bg-surface rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-surface-border">
                <h3 className="font-bold text-lg">How this risk is computed</h3>
                <button onClick={() => setExplain(null)} className="text-slate-400 hover:text-slate-200"><span className="material-symbols-outlined">close</span></button>
              </div>
              <div className="px-6 py-4 divide-y divide-slate-200 dark:divide-surface-border">
                <p className="pb-3 text-sm text-slate-600 dark:text-slate-300">
                  <b>{explain.model_name}</b> — Risk = <span className="text-primary font-semibold">Vulnerability</span> × <span className="font-semibold">Loss Magnitude</span>.
                </p>
                <Row label="Vulnerability">
                  <span className="text-primary font-bold">{mult.toFixed(0)}× random</span>
                  <span className="text-slate-400"> — measured by LeakPro (TPR@0.1%FPR).</span>
                  <div className="text-xs text-slate-400 mt-0.5">likelihood band L = <b className="text-slate-600 dark:text-slate-200">{v.L}</b> / 3</div>
                </Row>
                <Row label="Loss Magnitude">
                  <span className="text-slate-600 dark:text-slate-200">
                    sensitivity <b>{sensLabel(harm.sensitivity)}</b> ({v.sensW})
                    {harm.subjects >= 10000 ? " + large cohort (+1)" : ""} × subject type <b>{subjLabel(harm.subjectType)}</b> (×{v.subjW})
                  </span>
                  <div className="text-xs text-slate-400 mt-0.5">magnitude band S = <b className="text-slate-600 dark:text-slate-200">{v.S}</b> / 8</div>
                </Row>
                <Row label="Risk index">
                  <span className="font-mono text-slate-600 dark:text-slate-200">(L × S) / 24 × 100 = ({v.L} × {v.S}) / 24 × 100</span>
                  <div className="mt-1"><span className={`text-2xl font-black ${v.text}`}>{v.score}</span><span className="text-slate-400 text-xs"> / 100</span></div>
                </Row>
                <Row label="In practice">
                  <span className="text-slate-600 dark:text-slate-200">≈ <b>{v.expSubjects.toLocaleString()}</b> subjects re-identifiable</span>
                  <div className="text-xs text-slate-400 mt-0.5 font-mono">P({(v.p * 100).toFixed(2)}%) × {harm.subjects.toLocaleString()} subjects × {harm.records} rec.</div>
                </Row>
              </div>
              <div className="px-6 py-3 border-t border-slate-200 dark:border-surface-border text-[11px] text-slate-400 flex items-center gap-1">
                <span className="material-symbols-outlined text-xs">science</span>
                Transparent placeholder — the published model computes this as a FAIR Bayesian network (a distribution, not a point).
              </div>
            </div>
          </div>
        );
      })()}

      {/* ---- Wizard: step 2 inputs ---- */}
      {wizard === "inputs" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => setWizard(null)}>
          <div className="bg-white dark:bg-surface rounded-2xl shadow-2xl w-full max-w-xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-surface-border">
              <h3 className="font-bold text-lg">Your use case</h3>
              <button onClick={() => setWizard(null)} className="text-slate-400 hover:text-slate-200"><span className="material-symbols-outlined">close</span></button>
            </div>
            <div className="p-6 space-y-5">
              <div>
                <label className="text-xs text-slate-400">Data sensitivity
                  <InfoDot text="How harmful the leaked attribute is. Special-category (health, biometrics, ethnicity) is highest; contact info like an address is low." />
                </label>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {SENSITIVITY.map((s) => (
                    <button key={s.key} onClick={() => setDraft((d) => ({ ...d, sensitivity: s.key }))}
                      className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition-colors ${draft.sensitivity === s.key ? "bg-primary/15 border-primary text-primary" : "border-slate-300 dark:border-surface-border text-slate-500 hover:border-primary/50"}`}>{s.label}</button>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs text-slate-400">Data subject type
                  <InfoDot text="Who the data is about. Vulnerable groups (minors, patients) carry greater harm if exposed than the general population." />
                </label>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {SUBJECT_TYPES.map((s) => (
                    <button key={s.key} onClick={() => setDraft((d) => ({ ...d, subjectType: s.key }))}
                      className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition-colors ${draft.subjectType === s.key ? "bg-primary/15 border-primary text-primary" : "border-slate-300 dark:border-surface-border text-slate-500 hover:border-primary/50"}`}>{s.label}</button>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-slate-400">Subjects at risk
                    <InfoDot text="Number of individuals whose records are in the dataset. Scales the total harm — more people affected means larger magnitude." />
                  </label>
                  <input type="number" value={draft.subjects}
                    onChange={(e) => setDraft((d) => ({ ...d, subjects: Math.max(0, parseInt(e.target.value) || 0) }))}
                    className="mt-1 w-full px-3 py-2 rounded-lg bg-cream text-slate-900 border border-slate-300 dark:border-surface-border outline-none focus:border-primary font-mono text-sm" />
                </div>
                <div>
                  <label className="text-xs text-slate-400">Records / subject
                    <InfoDot text="How many records exist per subject. More records per person can increase identifiability and the harm from exposure." />
                  </label>
                  <input type="number" value={draft.records}
                    onChange={(e) => setDraft((d) => ({ ...d, records: Math.max(1, parseInt(e.target.value) || 1) }))}
                    className="mt-1 w-full px-3 py-2 rounded-lg bg-cream text-slate-900 border border-slate-300 dark:border-surface-border outline-none focus:border-primary font-mono text-sm" />
                </div>
              </div>
            </div>
            <div className="flex justify-between gap-3 px-6 py-4 border-t border-slate-200 dark:border-surface-border">
              <button onClick={() => setWizard("intro")} className="px-5 py-2 rounded-lg border border-slate-300 dark:border-surface-border text-sm font-bold hover:bg-slate-100 dark:hover:bg-surface-2 transition-colors">← Back</button>
              <button onClick={() => { setHarm(draft); setWizard(null); }} className="px-5 py-2 rounded-lg bg-slate-700 text-cream border border-primary text-sm font-bold hover:bg-slate-600 transition-colors">Calculate risk</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
