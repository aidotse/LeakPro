import React, { useState } from "react";
import { api, ModelResult, RiskAssessment, RiskRequest } from "../../api";
import InfoButton from "../InfoButton";
import MetaPanel from "./MetaPanel";
import RiskDiagram from "./RiskDiagram";

/**
 * Risk is never scored here. The measured half (TPR at the chosen operating point) comes straight from
 * the audit, and the harm half is declared by the user and combined by leakpro/risk on the backend.
 * This file only displays what it is given, which is why there is no threshold table in it.
 *
 * Presentation rule: the first view shows numbers, not prose. Every explanation lives behind an
 * InfoButton. The one exception is warnings — those stay visible as a chip, because a caveat like
 * "this operating point was not measurable" must not be something the reader has to go looking for.
 *
 * Risk state lives in the parent (Step7Results), not here. The tab bar unmounts this component when
 * the user switches tabs, so local state would silently discard a completed assessment.
 */

export interface RiskState {
  draft: RiskRequest;
  applied: RiskRequest | null;
  /** Keyed by `${job_id}/${model_name}` — model names are not unique across jobs in compare mode. */
  assessments: Record<string, RiskAssessment>;
  unassessable: Record<string, string>;
  error: string | null;
}

export const initialRiskState: RiskState = {
  draft: {
    tolerated_fpr: 0.01,
    attacker_prior: 0.5,
    records_per_subject: 1,
    data_type_sensitivity: 1,
    subject_type_weight: 1,
    extrapolate_to_population: false,
    notes: "",
  },
  applied: null,
  assessments: {},
  unassessable: {},
  error: null,
};

interface Props {
  results: ModelResult[];
  risk: RiskState;
  onRiskChange: (next: RiskState) => void;
}

function modelKey(m: ModelResult) {
  return `${m.job_id ?? ""}/${m.model_name}`;
}

const ALPHAS = [
  { value: 0.01, label: "1%" },
  { value: 0.001, label: "0.1%" },
  { value: 0.0001, label: "0.01%" },
];

const PRIORS = [
  { value: 0.5, label: "0.5 — balanced" },
  { value: 0.1, label: "0.1" },
  { value: 0.01, label: "0.01" },
  { value: 0.001, label: "0.001" },
];

// CNIL PIA-3 severity levels, offered as a starting scale. See leakpro/risk/policy.py.
const SENSITIVITY = [
  { value: 1, label: "1 — Negligible" },
  { value: 2, label: "2 — Limited" },
  { value: 3, label: "3 — Significant" },
  { value: 4, label: "4 — Maximum" },
];

const BAND_STYLE: Record<string, { color: string; bg: string }> = {
  SEVERE: { color: "text-red-500", bg: "bg-red-500/10 border border-red-500/30" },
  HIGH: { color: "text-orange-500", bg: "bg-orange-500/10 border border-orange-500/30" },
  MODERATE: { color: "text-amber-500", bg: "bg-amber-500/10 border border-amber-500/30" },
  LOW: { color: "text-green-500", bg: "bg-green-500/10 border border-green-500/30" },
  NONE: { color: "text-slate-400", bg: "bg-slate-100 dark:bg-surface-2" },
};

function bandStyle(band?: string) {
  return (band && BAND_STYLE[band]) || BAND_STYLE.NONE;
}

function pct(v: number | undefined | null) {
  return v != null ? (v * 100).toFixed(1) + "%" : "—";
}

function num(v: number | undefined | null, digits = 0) {
  return v != null ? v.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
}

/** Best TPR at a given operating point across a model's attacks, for the pre-assessment view. */
function bestTpr(model: ModelResult, alpha: number): { value: number; attack: string } | undefined {
  const key = alpha === 0.01 ? "TPR@1%FPR" : alpha === 0.001 ? "TPR@0.1%FPR" : "TPR@0.01%FPR";
  let best: { value: number; attack: string } | undefined;
  model.attacks.forEach((a) => {
    // Attacks with an inverted ROC are excluded, matching the backend's guard: an AUC below 0.5 means
    // a real leak would read as no leak.
    if (a.roc_auc != null && a.roc_auc < 0.5) return;
    const v = a.tpr_at_fpr?.[key];
    if (v != null && (best === undefined || v > best.value)) best = { value: v, attack: a.attack_name };
  });
  return best;
}

// ---------------------------------------------------------------------------
// Explanation content. Kept here, out of the layout, so the panels stay readable.
// ---------------------------------------------------------------------------

const HOW_IT_WORKS = (
  <>
    <p>
      The audit measures how often an attack correctly identifies a training member. Whether that
      matters depends on your deployment, which LeakPro cannot observe. So the assessment keeps the two
      halves apart and combines them without inventing any weights.
    </p>
    <RiskDiagram className="my-1" />
    <p>
      <b>Nothing is hidden in a coefficient.</b> Every number is either measured by this audit or
      declared by you, and the result lists the assumption behind each derived figure.
    </p>
    <p>
      <b>Why Loss Event Frequency is just the measured success rate.</b> In the published model,
      <code> LEF = V × (RP × TEF)</code>, where <b>RP</b> is the <i>retention period</i> — how long the
      data stays available to be attacked — and <b>TEF</b> is the <i>threat event frequency</i>, how
      often an adversary tries. LeakPro can observe neither, so both are set to 1: a single attack
      attempt against a model that is retained. If repeated attempts are plausible in your setting,
      the real frequency is higher than this assessment assumes, and you should scale it up yourself.
    </p>
    <p className="text-xs text-slate-400">
      Structure: Sion, Van Landuyt, Wuyts &amp; Joosen, “Privacy Risk Assessment for Data
      Subject-aware Threat Modeling”, IWPE 2019. Precision: Jayaraman, Wang, Knipmeyer, Gu &amp; Evans,
      “Revisiting Membership Inference Under Realistic Assumptions”, PoPETs 2021, Theorem 4.2.
      Sensitivity scale: CNIL PIA-3 knowledge bases, 2018.
    </p>
  </>
);

const ABOUT_ALPHA = (
  <>
    <p>
      α is the false-positive rate the attacker is assumed to tolerate. It sets the operating point the
      whole assessment is read at, so there is no default: the choice is yours.
    </p>
    <ul className="list-disc ml-5 flex flex-col gap-1">
      <li><b>1%</b> — resolvable on the audit set sizes most targets have. Start here.</li>
      <li><b>0.1%</b> — stricter. Needs at least 1000 non-members to be measurable at all.</li>
      <li><b>0.01%</b> — very strict. Needs at least 10000 non-members.</li>
    </ul>
    <p>
      Below the resolution of your audit set, a true positive rate of zero means “not measurable”, not
      “no leakage”. The backend refuses to report derived figures there rather than let you read a zero
      as safety.
    </p>
  </>
);

const ABOUT_PRIOR = (
  <>
    <p>
      π is the share of the attacker’s candidate pool that really are members. It does not change the
      measurement; it changes what the measurement means.
    </p>
    <p>
      The audit runs on a balanced split, so every TPR and AUC it reports implicitly assumes π = 0.5.
      Real pools are usually far more skewed, and precision falls steeply as they are. At a 1%
      false-positive rate and a measured TPR of 10%, an attacker is right 91% of the time at π = 0.5 —
      and about 1% of the time at π = 0.001.
    </p>
    <p>Both values are always reported, so a balanced-prior figure can never be quoted by accident.</p>
  </>
);

const ABOUT_FACTORS = (
  <>
    <p>
      These four are the Loss Magnitude factors from Sion et al.: <code>LM = DTS × NR × DST × NDS</code>.
      The paper deliberately supplies no numeric values for them, so they are yours to set. All default
      to a neutral 1.0, which reduces the loss magnitude to a plain count of subjects.
    </p>
    <ul className="list-disc ml-5 flex flex-col gap-1">
      <li><b>DTS</b> — sensitivity of the leaked data type. The CNIL PIA severity levels are a defensible starting scale.</li>
      <li><b>NR</b> — records of this data type per subject. Fractions are fine when only some subjects contribute it.</li>
      <li><b>DST</b> — weight for the subject type. Raise it for vulnerable subjects such as minors or patients.</li>
      <li><b>NDS</b> — number of data subjects. Left blank, the target’s training-set size is used.</li>
    </ul>
    <p className="text-xs text-slate-400">
      Sion et al. note the model assumes these factors are independent, which they flag as a
      simplification of reality. That assumption is recorded in every assessment.
    </p>
  </>
);

const ABOUT_TPR = (
  <p>
    Of all the training members in the audit set, the share the strongest attack correctly identifies
    while keeping its false-positive rate at or below α. Attacks whose ROC is inverted (AUC below 0.5)
    are excluded rather than counted as weak evidence.
  </p>
);

const ABOUT_LIFT = (
  <p>
    How many times better than random guessing the attack is at this operating point: TPR divided by α.
    A lift of 1× is no better than chance. The band label is derived from this figure alone, and it is
    an unsourced convenience — no published thresholds exist for it.
  </p>
);

const ABOUT_PPV = (
  <>
    <p>
      If the attacker claims a record was in the training set, how often are they right? This is the
      number that decides whether a leak is actionable, and it depends on your declared prior π as much
      as on the attack.
    </p>
    <p className="font-mono text-xs">PPV = TPR / (TPR + γ·α), γ = (1 − π) / π</p>
    <p>
      <b>γ is how outnumbered the members are:</b> for every genuine member in the attacker’s candidate
      pool there are γ non-members. It is derived from your π, not a separate input — π = 0.5 gives
      γ = 1, π = 0.01 gives γ = 99, π = 0.001 gives γ = 999.
    </p>
    <p>
      That is why it multiplies the false-positive rate. At π = 0.001 with a measured TPR of 10% and
      α = 1%, each member yields 0.10 expected true positives while the 999 non-members yield
      999 × 0.01 = 9.99 false ones — about ten false alarms per correct hit, so precision is roughly
      1%. The identical measurement reads 91% at π = 0.5.
    </p>
    <p className="text-xs text-slate-400">
      A tenfold more skewed pool costs as much precision as a tenfold looser threshold, which is the
      same reason strict operating points matter.
    </p>
  </>
);

const ABOUT_EXPOSED = (
  <>
    <p>
      The measured success rate applied to your declared population: how many subjects an attacker would
      be expected to identify. Sensitivity weights are deliberately left out of this figure so it stays
      a plain count you can reason about.
    </p>
    <p>
      <b>This count does not depend on π, and should not.</b> The true positive rate is conditional on a
      record actually being a member, so it is unaffected by how the attacker’s pool is composed —
      which means this figure already counts true positives only. Multiplying it by precision would be
      circular, since precision times the flagged set <i>is</i> the true-positive count.
    </p>
    <p>
      A skewed prior does not reduce how many members are genuinely identified; it increases the false
      accusations sitting alongside them. So read the two columns as different questions: this one is
      how many people the attack really exposes, precision is how far an attacker can trust any single
      claim. “1000 exposed at 1% precision” and “40 exposed at 99% precision” are both real findings,
      and they call for different responses.
    </p>
  </>
);

const ABOUT_BAND = (
  <p>
    An advisory label over the measured lift only, never over the combination of measurement and
    judgement. The thresholds are round numbers chosen so that NONE means no better than chance and
    SEVERE means two orders of magnitude better than chance. They are not calibrated against anything
    published, which is why the policy version travels with every assessment.
  </p>
);

export default function Summary({ results, risk, onRiskChange }: Props) {
  // Ephemeral view state only. Anything the user would be annoyed to lose lives in `risk`.
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [wizard, setWizard] = useState(false);
  const [busy, setBusy] = useState(false);
  const [details, setDetails] = useState<string | null>(null);

  const { draft, applied, assessments, unassessable, error } = risk;
  const setDraft = (next: RiskRequest) => onRiskChange({ ...risk, draft: next });

  const alpha = applied?.tolerated_fpr ?? draft.tolerated_fpr;
  const toggleInfo = (key: string) => setOpenKey((prev) => (prev === key ? null : key));

  const submit = async () => {
    // Compare mode can show models from several jobs, and the endpoint is per job, so assess each job
    // the visible models come from and merge under composite keys.
    const jobIds = Array.from(new Set(results.map((m) => m.job_id).filter((v): v is string => !!v)));
    if (!jobIds.length) {
      onRiskChange({ ...risk, error: "No job id on these results." });
      return;
    }
    setBusy(true);
    onRiskChange({ ...risk, error: null });
    try {
      const responses = await Promise.all(jobIds.map((id) => api.assessRisk(id, draft)));
      const assessed: Record<string, RiskAssessment> = {};
      const failed: Record<string, string> = {};
      responses.forEach((res, i) => {
        Object.entries(res.assessments).forEach(([name, a]) => { assessed[`${jobIds[i]}/${name}`] = a; });
        Object.entries(res.unassessable).forEach(([name, why]) => { failed[`${jobIds[i]}/${name}`] = why; });
      });
      onRiskChange({ ...risk, assessments: assessed, unassessable: failed, applied: draft, error: null });
      setWizard(false);
    } catch (e) {
      onRiskChange({ ...risk, error: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  const downloadCSV = () => {
    const header = [
      "model", "model_class", "alpha", "tpr_at_alpha", "advantage", "lift", "roc_auc",
      "train_accuracy", "test_accuracy", "dpsgd", "attacker_prior", "ppv", "ppv_at_balanced_prior",
      "n_subjects", "loss_magnitude", "risk_lm_x_lef", "expected_exposed_subjects", "expected_cost",
      "vulnerability_band", "policy_version",
    ];
    const rows = [header.join(",")];
    results.forEach((m) => {
      const a = assessments[modelKey(m)];
      const tpr = bestTpr(m, alpha);
      rows.push([
        m.model_name,
        m.model_class ?? "",
        alpha,
        a?.measured.success_rate?.toFixed(6) ?? tpr?.value.toFixed(6) ?? "",
        a?.measured.advantage?.toFixed(6) ?? "",
        a?.measured.lift?.toFixed(3) ?? "",
        a?.measured.roc_auc?.toFixed(6) ?? "",
        m.train_accuracy?.toFixed(6) ?? "",
        m.test_accuracy?.toFixed(6) ?? "",
        m.dpsgd ? `eps=${m.target_epsilon}` : "No",
        a?.declared.attacker_prior ?? "",
        a?.combined.ppv?.toFixed(6) ?? "",
        a?.combined.ppv_balanced?.toFixed(6) ?? "",
        a?.declared.n_subjects ?? "",
        a?.combined.loss_magnitude?.toFixed(2) ?? "",
        a?.combined.risk?.toFixed(2) ?? "",
        a?.combined.expected_exposed_subjects?.toFixed(2) ?? "",
        a?.combined.expected_cost?.toFixed(2) ?? "",
        a?.combined.vulnerability_band ?? "",
        a?.policy_version ?? "",
      ].join(","));
    });
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const el = document.createElement("a"); el.href = url;
    el.download = "leakpro_summary.csv"; el.click();
    URL.revokeObjectURL(url);
  };

  const alphaLabel = ALPHAS.find((a) => a.value === alpha)?.label ?? `${alpha * 100}%`;
  const shown = details ? assessments[details] : undefined;

  return (
    <div className="flex flex-col gap-8">
      {/* Cards: measured lift until a use case is declared, band afterwards */}
      {results.length > 1 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {results.map((m) => {
            const a = assessments[modelKey(m)];
            const tpr = bestTpr(m, alpha);
            const style = bandStyle(a?.combined.vulnerability_band);
            return (
              <div key={modelKey(m)}
                   className={a ? `rounded-xl p-4 ${style.bg}` : "rounded-xl p-4 bg-slate-50 dark:bg-surface border border-slate-200 dark:border-surface-border"}>
                <p className={`text-2xl font-black ${a ? style.color : "text-primary"}`}>
                  {a ? a.combined.vulnerability_band : `${num(tpr ? tpr.value / alpha : undefined, 1)}×`}
                </p>
                <p className="text-[11px] uppercase tracking-wider text-slate-400 -mt-0.5">
                  {a ? "vulnerability" : "vs random"}
                </p>
                <p className="font-bold text-sm mt-2">{m.model_name}</p>
                {m.model_class && <p className="text-xs font-mono text-slate-400 mt-0.5">{m.model_class}</p>}
                {m.dpsgd && <p className="text-xs text-blue-500 mt-1">DP-SGD{m.target_epsilon != null ? ` ε=${m.target_epsilon}` : ""}</p>}
              </div>
            );
          })}
        </div>
      )}

      {/* Risk bar. One line of text, everything else behind the info button. */}
      <div className="flex items-center justify-between gap-3 flex-wrap rounded-xl border border-slate-200 dark:border-surface-border bg-slate-50 dark:bg-surface px-4 py-3">
        <div>
          <p className="font-bold text-sm flex items-center gap-1">
            Risk assessment
            <InfoButton label="How risk is computed" title="How risk is computed">{HOW_IT_WORKS}</InfoButton>
          </p>
          <p className="text-xs text-slate-400">
            {applied
              ? `α = ${alphaLabel} FPR · π = ${applied.attacker_prior} · policy ${Object.values(assessments)[0]?.policy_version ?? "—"}`
              : "Measured leakage is above. Add your use case to turn it into risk."}
          </p>
        </div>
        <button
          onClick={() => setWizard(true)}
          className="px-3 py-1.5 rounded-lg bg-primary text-white text-xs font-bold hover:opacity-90 transition-opacity"
        >
          {applied ? "Change inputs" : "Declare your use case"}
        </button>
      </div>

      {error && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-500">{error}</div>
      )}

      {Object.entries(unassessable).length > 0 && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm">
          <p className="font-bold text-amber-500 mb-1">Not assessable</p>
          <ul className="list-disc ml-5 text-slate-600 dark:text-slate-300">
            {Object.entries(unassessable).map(([key, reason]) => (
              <li key={key}><b>{key.slice(key.indexOf("/") + 1)}</b>: {reason}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Comparison table */}
      <div className="rounded-xl border border-slate-200 dark:border-surface-border overflow-hidden overflow-x-auto">
        <div className="flex items-center justify-between px-3 py-2 bg-slate-50 dark:bg-surface border-b border-slate-200 dark:border-surface-border">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Summary</span>
          <button
            onClick={downloadCSV}
            className="flex items-center gap-1 px-2 py-1 rounded border border-slate-300 dark:border-surface-border text-slate-500 text-xs font-bold hover:bg-slate-100 dark:hover:bg-surface-2 transition-colors"
          >
            <span className="material-symbols-outlined text-sm">download</span>
            CSV
          </button>
        </div>
        <table className="w-full text-sm">
          <thead className="bg-slate-50 dark:bg-surface border-b border-slate-200 dark:border-surface-border">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">Model</th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">
                TPR@{alphaLabel} <InfoButton label={`TPR at ${alphaLabel} FPR`}>{ABOUT_TPR}</InfoButton>
              </th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">
                Lift <InfoButton label="Lift over random guessing">{ABOUT_LIFT}</InfoButton>
              </th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">
                Precision <InfoButton label="Attacker precision (PPV)">{ABOUT_PPV}</InfoButton>
              </th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">
                Exposed <InfoButton label="Expected exposed subjects">{ABOUT_EXPOSED}</InfoButton>
              </th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">DP-SGD</th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">
                Band <InfoButton label="Vulnerability band">{ABOUT_BAND}</InfoButton>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-surface-border">
            {results.map((m) => {
              const key = modelKey(m);
              const a = assessments[key];
              const tpr = bestTpr(m, alpha);
              const style = bandStyle(a?.combined.vulnerability_band);
              const isOpen = openKey === key;
              return (
                <React.Fragment key={key}>
                  <tr className="hover:bg-slate-50/50 dark:hover:bg-surface/50 transition-colors">
                    <td className="px-4 py-3 font-semibold">
                      <div className="flex items-center gap-2 flex-wrap">
                        {m.model_name}
                        {m.model_class && (
                          <span className="text-xs px-1.5 py-0.5 rounded bg-slate-100 dark:bg-surface-2 text-slate-500 font-mono">
                            {m.model_class}
                          </span>
                        )}
                        <button
                          onClick={() => toggleInfo(key)}
                          title="Show model info"
                          className={`material-symbols-outlined text-base transition-colors ${isOpen ? "text-primary" : "text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"}`}
                        >
                          info
                        </button>
                        {a && (
                          <button
                            onClick={() => setDetails(key)}
                            title="How this was computed"
                            className="material-symbols-outlined text-base text-slate-400 hover:text-primary transition-colors"
                          >
                            calculate
                          </button>
                        )}
                        {a && a.warnings.length > 0 && (
                          <button
                            onClick={() => setDetails(key)}
                            title="Show caveats"
                            className="px-1.5 py-0.5 rounded text-[11px] font-bold bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30 hover:bg-amber-500/25 transition-colors"
                          >
                            {a.warnings.length} caveat{a.warnings.length > 1 ? "s" : ""}
                          </button>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono">
                      {tpr !== undefined
                        ? <>{pct(tpr.value)} <span className="text-slate-400 font-sans text-xs">({tpr.attack})</span></>
                        : "—"}
                    </td>
                    <td className="px-4 py-3 font-mono">{tpr ? `${num(tpr.value / alpha, 1)}×` : "—"}</td>
                    <td className="px-4 py-3 font-mono">
                      {a?.combined.ppv != null
                        ? <>{pct(a.combined.ppv)} <span className="text-slate-400 font-sans text-xs">(π={a.declared.attacker_prior})</span></>
                        : <span className="text-slate-400 font-sans text-xs">—</span>}
                    </td>
                    <td className="px-4 py-3 font-mono">
                      {a?.combined.expected_exposed_subjects != null ? num(a.combined.expected_exposed_subjects, 1) : "—"}
                    </td>
                    <td className="px-4 py-3">
                      {m.dpsgd
                        ? <span className="text-blue-500 font-semibold">{m.target_epsilon != null ? `ε=${m.target_epsilon}` : "Yes"}</span>
                        : <span className="text-slate-400">No</span>}
                    </td>
                    <td className="px-4 py-3">
                      {a?.combined.vulnerability_band
                        ? <span className={`font-bold ${style.color}`}>{a.combined.vulnerability_band}</span>
                        : <span className="text-slate-400">—</span>}
                    </td>
                  </tr>
                  {isOpen && (
                    <tr className="bg-slate-50 dark:bg-surface/60">
                      <td colSpan={7} className="px-6 py-4"><MetaPanel r={m} /></td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* One sentence per model. Detail is a click away, not on the page. */}
      <div className="flex flex-col gap-3">
        {results.map((m) => {
          const key = modelKey(m);
          const a = assessments[key];
          const tpr = bestTpr(m, alpha);
          const style = bandStyle(a?.combined.vulnerability_band);
          return (
            <div key={key}
                 className={a ? `rounded-xl p-5 ${style.bg}` : "rounded-xl p-5 bg-slate-50 dark:bg-surface border border-slate-200 dark:border-surface-border"}>
              <p className={`font-bold mb-1 ${a ? style.color : ""}`}>
                {m.model_name}
                {m.model_class && <span className="ml-2 text-xs font-mono text-slate-400">{m.model_class}</span>}
                {a?.combined.vulnerability_band ? ` — ${a.combined.vulnerability_band} vulnerability` : ""}
              </p>
              <p className="text-sm text-slate-600 dark:text-slate-300">
                {tpr
                  ? `At a ${alphaLabel} false-positive rate, ${tpr.attack} identifies ${pct(tpr.value)} of training members, ${num(tpr.value / alpha, 1)}× better than chance.`
                  : "No usable attack results at this operating point."}
                {a?.combined.ppv != null && ` An attacker claiming membership would be right ${pct(a.combined.ppv)} of the time.`}
              </p>
              {a && (
                <button
                  onClick={() => setDetails(key)}
                  className="mt-2 text-xs font-bold text-primary hover:underline"
                >
                  Show the numbers behind this
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* Wizard: labels and inputs only, explanations behind info buttons */}
      {wizard && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 p-4" onClick={() => setWizard(false)}>
          <div className="max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-2xl bg-white dark:bg-surface-deep p-6 shadow-2xl"
               onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-black mb-1 flex items-center gap-1">
              Your use case
              <InfoButton label="How risk is computed" title="How risk is computed">{HOW_IT_WORKS}</InfoButton>
            </h3>
            <p className="text-xs text-slate-500 mb-5">Only you can supply these. Defaults are neutral.</p>

            <div className="flex flex-col gap-4 text-sm">
              <label className="flex flex-col gap-1">
                <span className="font-bold flex items-center gap-1">
                  Tolerated false-positive rate (α)
                  <InfoButton label="Operating point (α)">{ABOUT_ALPHA}</InfoButton>
                </span>
                <select
                  className="rounded-lg border border-slate-300 dark:border-surface-border bg-transparent px-3 py-2"
                  value={draft.tolerated_fpr}
                  onChange={(e) => setDraft({ ...draft, tolerated_fpr: Number(e.target.value) })}
                >
                  {ALPHAS.map((a) => <option key={a.value} value={a.value}>{a.label}</option>)}
                </select>
              </label>

              <label className="flex flex-col gap-1">
                <span className="font-bold flex items-center gap-1">
                  Attacker prior (π)
                  <InfoButton label="Attacker prior (π)">{ABOUT_PRIOR}</InfoButton>
                </span>
                <select
                  className="rounded-lg border border-slate-300 dark:border-surface-border bg-transparent px-3 py-2"
                  value={draft.attacker_prior}
                  onChange={(e) => setDraft({ ...draft, attacker_prior: Number(e.target.value) })}
                >
                  {PRIORS.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
                </select>
              </label>

              <div className="flex items-center gap-1">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Harm factors</span>
                <InfoButton label="Harm factors" title="Harm factors (Loss Magnitude)">{ABOUT_FACTORS}</InfoButton>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <label className="flex flex-col gap-1">
                  <span className="font-bold">Data subjects (NDS)</span>
                  <input
                    type="number" min={1} placeholder="training-set size"
                    className="rounded-lg border border-slate-300 dark:border-surface-border bg-transparent px-3 py-2"
                    value={draft.n_subjects ?? ""}
                    onChange={(e) => setDraft({ ...draft, n_subjects: e.target.value ? Number(e.target.value) : undefined })}
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="font-bold">Records per subject (NR)</span>
                  <input
                    type="number" min={0.01} step={0.01}
                    className="rounded-lg border border-slate-300 dark:border-surface-border bg-transparent px-3 py-2"
                    value={draft.records_per_subject}
                    onChange={(e) => setDraft({ ...draft, records_per_subject: Number(e.target.value) })}
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="font-bold">Sensitivity (DTS)</span>
                  <select
                    className="rounded-lg border border-slate-300 dark:border-surface-border bg-transparent px-3 py-2"
                    value={draft.data_type_sensitivity}
                    onChange={(e) => setDraft({ ...draft, data_type_sensitivity: Number(e.target.value) })}
                  >
                    {SENSITIVITY.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="font-bold">Subject weight (DST)</span>
                  <input
                    type="number" min={0.01} step={0.5}
                    className="rounded-lg border border-slate-300 dark:border-surface-border bg-transparent px-3 py-2"
                    value={draft.subject_type_weight}
                    onChange={(e) => setDraft({ ...draft, subject_type_weight: Number(e.target.value) })}
                  />
                </label>
              </div>

              <label className="flex flex-col gap-1">
                <span className="font-bold flex items-center gap-1">
                  Cost per exposed subject
                  <InfoButton label="Cost per exposed subject">
                    <p>
                      Optional, in any unit you choose. Left blank, no monetary figure is reported at
                      all — a default cost would be fabricated, and a fabricated cost is worse than
                      none.
                    </p>
                  </InfoButton>
                </span>
                <input
                  type="number" min={0} step={1} placeholder="optional"
                  className="rounded-lg border border-slate-300 dark:border-surface-border bg-transparent px-3 py-2"
                  value={draft.cost_per_exposed_subject ?? ""}
                  onChange={(e) => setDraft({ ...draft, cost_per_exposed_subject: e.target.value ? Number(e.target.value) : undefined })}
                />
              </label>

              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={draft.extrapolate_to_population ?? false}
                  onChange={(e) => setDraft({ ...draft, extrapolate_to_population: e.target.checked })}
                />
                <span className="flex items-center gap-1">
                  Extrapolate exposed records to the full training set
                  <InfoButton label="Extrapolation">
                    <p>
                      Scales the exposed-record count from the audit set up to the training population.
                      Off by default: per-record vulnerability is strongly non-uniform, so this is an
                      estimate under an assumption, not a measurement.
                    </p>
                  </InfoButton>
                </span>
              </label>

              <label className="flex flex-col gap-1">
                <span className="font-bold">Notes</span>
                <textarea
                  rows={2} placeholder="kept in the audit trail"
                  className="rounded-lg border border-slate-300 dark:border-surface-border bg-transparent px-3 py-2"
                  value={draft.notes ?? ""}
                  onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
                />
              </label>
            </div>

            <div className="mt-6 flex justify-end gap-2">
              <button onClick={() => setWizard(false)}
                      className="px-4 py-2 rounded-lg border border-slate-300 dark:border-surface-border text-sm font-bold">
                Cancel
              </button>
              <button onClick={submit} disabled={busy}
                      className="px-4 py-2 rounded-lg bg-primary text-white text-sm font-bold disabled:opacity-50">
                {busy ? "Assessing…" : "Assess risk"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Details: formula, assumptions, caveats */}
      {details && shown && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 p-4" onClick={() => setDetails(null)}>
          <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-2xl bg-white dark:bg-surface-deep p-6 shadow-2xl"
               onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-black mb-1">
              {details.slice(details.indexOf("/") + 1)}
            </h3>
            <p className="text-xs text-slate-500 mb-4">
              Policy {shown.policy_version} · LeakPro {shown.leakpro_version}
            </p>

            <RiskDiagram className="mb-4" />

            <pre className="text-xs bg-slate-100 dark:bg-surface-2 rounded-lg p-3 overflow-x-auto mb-4">
{`LEF  = measured TPR at α    = ${shown.combined.loss_event_frequency}
LM   = DTS × NR × DST × NDS = ${shown.combined.loss_magnitude ?? "—"}
Risk = LEF × LM             = ${shown.combined.risk ?? "—"}
PPV  = TPR / (TPR + γ·α)    = ${shown.combined.ppv ?? "—"}   γ = ${shown.combined.gamma}`}
            </pre>

            {shown.warnings.length > 0 && (
              <>
                <p className="font-bold text-sm mb-1 text-amber-500">Caveats</p>
                <ul className="list-disc ml-5 text-xs text-amber-600 dark:text-amber-400 flex flex-col gap-1 mb-4">
                  {shown.warnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              </>
            )}

            <p className="font-bold text-sm mb-1">Assumptions</p>
            <ol className="list-decimal ml-5 text-xs text-slate-600 dark:text-slate-300 flex flex-col gap-1">
              {shown.assumptions.map((a, i) => <li key={i}>{a}</li>)}
            </ol>

            <div className="mt-6 flex justify-end">
              <button onClick={() => setDetails(null)}
                      className="px-4 py-2 rounded-lg bg-primary text-white text-sm font-bold">Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
