import React, { useState } from "react";
import { api, ModelResult, RiskAssessment, RiskRequest } from "../../api";
import InfoButton, { Detail } from "../InfoButton";
import MetaPanel from "./MetaPanel";
import { COPY } from "./optimizationCopy";
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
  /** Opens the optimization view for one model. Omitted when unavailable. */
  onOptimize?: (model: ModelResult) => void;
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
// Explanation content. Each one opens with a plain-language lead; the depth sits behind a nested
// Detail disclosure so the first thing a reader sees is never a wall of text.
// ---------------------------------------------------------------------------

const HOW_IT_WORKS = (
  <>
    <p>
      The audit measures how often an attack succeeds. You say how much a success would cost. Risk is
      those two multiplied.
    </p>
    <RiskDiagram className="my-1" />
    <Detail>
      <p>
        Nothing is hidden in a coefficient: every number is either measured here or declared by you, and
        the result lists the assumption behind each derived figure.
      </p>
      <p>
        Loss Event Frequency is the measured success rate on its own because the published model puts two
        more factors in it — retention period (how long the data stays attackable) and threat event
        frequency (how often an adversary tries). LeakPro can observe neither, so both are set to 1: one
        attempt on a model that is kept. If repeated attempts are plausible for you, real frequency is
        higher than this assumes.
      </p>
      <p className="text-xs text-slate-400">
        Structure: Sion, Van Landuyt, Wuyts &amp; Joosen, IWPE 2019. Precision: Jayaraman, Wang,
        Knipmeyer, Gu &amp; Evans, PoPETs 2021, Theorem 4.2. Sensitivity scale: CNIL PIA-3, 2018.
      </p>
    </Detail>
  </>
);

const ABOUT_ALPHA = (
  <>
    <p>How many false alarms the attacker puts up with. Lower is stricter. Use 1% unless you have a reason not to.</p>
    <Detail>
      <p>
        A stricter rate needs a bigger audit set to be measurable at all: you need at least 1/α
        non-members, so 1000 for 0.1% and 10000 for 0.01%.
      </p>
      <p>
        Below that, a true positive rate of zero means “not measurable”, not “no leakage”, so the
        derived figures are withheld rather than letting a zero read as safety.
      </p>
    </Detail>
  </>
);

const ABOUT_PRIOR = (
  <>
    <p>
      How common members are in the pool the attacker is guessing from. The audit assumes half of them
      are, which makes any attack look better than it would in practice.
    </p>
    <Detail>
      <p>
        It changes what the measurement means, not the measurement. With a 10% true positive rate at a 1%
        false-alarm rate, an attacker is right 91% of the time if half the pool are members, and about 1%
        of the time if one in a thousand are.
      </p>
      <p>Both values are always reported, so the flattering one can never be quoted by accident.</p>
    </Detail>
  </>
);

const ABOUT_FACTORS = (
  <>
    <p>
      Four numbers describing how bad a leak would be for the people in your data. All default to 1,
      meaning no weighting.
    </p>
    <Detail>
      <ul className="list-disc ml-5 flex flex-col gap-1">
        <li><b>Sensitivity</b> — how revealing the leaked data type is. The CNIL PIA severity levels are a defensible scale.</li>
        <li><b>Records per subject</b> — fractions are fine when only some subjects contribute the data.</li>
        <li><b>Subject weight</b> — raise it for vulnerable subjects such as minors or patients.</li>
        <li><b>Data subjects</b> — left blank, the target’s training-set size is used.</li>
      </ul>
      <p>
        These are the Loss Magnitude factors of Sion et al., who supply the structure but deliberately no
        values, and who note the model treats the factors as independent — a simplification recorded in
        every assessment.
      </p>
    </Detail>
  </>
);

const ABOUT_TPR = (
  <>
    <p>Share of training members the best attack finds, at your false-alarm rate.</p>
    <Detail>
      <p>
        Attacks whose ROC is inverted (AUC below 0.5) are excluded rather than counted as weak evidence,
        because an inverted result makes a real leak read as no leak.
      </p>
    </Detail>
  </>
);

const ABOUT_LIFT = (
  <>
    <p>How many times better than guessing the attack is. 1× is chance.</p>
    <Detail>
      <p>
        True positive rate divided by the false-alarm rate. The band label comes from this figure alone,
        and its thresholds are round numbers rather than anything published.
      </p>
    </Detail>
  </>
);

const ABOUT_PPV = (
  <>
    <p>When the attacker says “this record was used in training”, how often they are right.</p>
    <Detail>
      <p className="font-mono text-xs">PPV = TPR / (TPR + γ·α), γ = (1 − π) / π</p>
      <p>
        γ is how outnumbered the members are: for every real member in the pool there are γ others. It
        comes from your prior, so π = 0.5 gives γ = 1 and π = 0.001 gives γ = 999.
      </p>
      <p>
        That is why it multiplies the false-alarm rate. At π = 0.001 with a 10% true positive rate and a
        1% false-alarm rate, one member yields 0.10 correct flags while the other 999 yield about 10
        wrong ones — so roughly ten false alarms per hit.
      </p>
    </Detail>
  </>
);

const ABOUT_EXPOSED = (
  <>
    <p>Roughly how many of your subjects an attacker would actually identify.</p>
    <Detail>
      <p>
        This count does not depend on the prior, and should not. The true positive rate is conditional on
        a record really being a member, so the figure already counts correct identifications only.
        Multiplying it by precision would be circular, since precision times the flagged set is that same
        count.
      </p>
      <p>
        A skewed prior does not reduce how many members are identified; it adds false accusations beside
        them. So read the two columns as different questions: this one is how many people are exposed,
        precision is how far a single claim can be trusted.
      </p>
      <p>Sensitivity weights are left out on purpose, so this stays a plain count.</p>
    </Detail>
  </>
);

const ABOUT_BAND = (
  <>
    <p>A rough label for the measured lift. Advisory only.</p>
    <Detail>
      <p>
        It labels the measurement, never the combination of measurement and judgement. NONE means no
        better than chance and SEVERE means two orders of magnitude better, but the thresholds are not
        calibrated against anything published — which is why the policy version travels with every
        assessment.
      </p>
    </Detail>
  </>
);

export default function Summary({ results, onOptimize, risk, onRiskChange }: Props) {
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

      {/* Protection — one entry per model, since each is optimized on its own */}
      {onOptimize && results.length > 0 && (
        <div className="rounded-xl border border-slate-200 dark:border-surface-border p-4 flex flex-col gap-3">
          <div>
            <p className="font-bold text-sm">Reduce the risk</p>
            <p className="text-xs text-slate-500 mt-0.5">
              Test protection settings automatically and see the best trade-offs between privacy and quality.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {results.map((m) => (
              <button
                key={`${m.job_id}/${m.model_name}`}
                onClick={() => onOptimize(m)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg border border-primary/50 text-primary text-sm font-bold hover:bg-primary/5 transition-colors"
              >
                <span className="material-symbols-outlined text-base">shield</span>
                {COPY.entry}
                {results.length > 1 && (
                  <span className="font-normal text-slate-400">· {m.model_name}</span>
                )}
              </button>
            ))}
          </div>
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
                        {m.optimized && (
                          <span className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary font-bold uppercase tracking-wider">
                            optimized
                          </span>
                        )}
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
                    <p>Optional, in any unit you like.</p>
                    <Detail>
                      <p>
                        Left blank, no monetary figure is reported at all. A default cost would be made
                        up, and a made-up cost is worse than none.
                      </p>
                    </Detail>
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
                    <p>Scales the exposed count from the audit set up to the whole training set.</p>
                    <Detail>
                      <p>
                        Off by default, because some records are far more exposed than others. The
                        scaled number is an estimate under that assumption, not a measurement.
                      </p>
                    </Detail>
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
