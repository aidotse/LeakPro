import React, { useState } from "react";
import { ModelResult, TrainMeta } from "../../api";

interface Props { results: ModelResult[] }

function riskLevel(auc: number | undefined): { label: string; color: string; bg: string } {
  if (auc === undefined) return { label: "N/A", color: "text-slate-400", bg: "bg-slate-100 dark:bg-slate-800" };
  if (auc >= 0.75)        return { label: "HIGH",   color: "text-red-500",    bg: "bg-red-500/10 border border-red-500/30" };
  if (auc >= 0.60)        return { label: "MEDIUM", color: "text-orange-500", bg: "bg-orange-500/10 border border-orange-500/30" };
  return                         { label: "LOW",    color: "text-green-500",  bg: "bg-green-500/10 border border-green-500/30" };
}

function bestAuc(model: ModelResult): number | undefined {
  const vals = model.attacks.map((a) => a.roc_auc).filter((v): v is number => v !== null && v !== undefined);
  return vals.length ? Math.max(...vals) : undefined;
}

function bestTpr(model: ModelResult): number | undefined {
  const vals = model.attacks
    .map((a) => a.tpr_at_fpr?.["TPR@0.1%FPR"])
    .filter((v): v is number => v !== null && v !== undefined);
  return vals.length ? Math.max(...vals) : undefined;
}

function pct(v: number | undefined) {
  return v !== undefined ? (v * 100).toFixed(1) + "%" : "—";
}

function MetaPanel({ r }: { r: ModelResult }) {
  const m = r.train_meta;
  const rows: { label: string; value: string }[] = [];

  if (r.model_class) rows.push({ label: "Architecture", value: r.model_class });

  if (r.dpsgd) {
    const dp = [`DP-SGD  ε=${r.target_epsilon ?? "?"}`];
    if (m?.target_delta != null) dp.push(`δ=${m.target_delta}`);
    if (m?.max_grad_norm != null) dp.push(`clip=${m.max_grad_norm}`);
    if (m?.virtual_batch_size != null) dp.push(`vbs=${m.virtual_batch_size}`);
    rows.push({ label: "Privacy", value: dp.join("  ") });
  } else {
    rows.push({ label: "Privacy", value: "Standard training (no DP)" });
  }

  if (m) {
    if (m.optimizer || m.learning_rate != null || m.batch_size != null) {
      const opt = [m.optimizer ?? "—", `lr=${m.learning_rate ?? "?"}`, `batch=${m.batch_size ?? "?"}`].join("  ");
      rows.push({ label: "Optimizer", value: opt });
    }
    if (m.epochs != null) rows.push({ label: "Epochs", value: String(m.epochs) });
    if (m.f_train != null || m.f_test != null) {
      rows.push({ label: "Data split", value: `train=${pct(m.f_train)}  test=${pct(m.f_test)}` });
    }
    const dataParts: string[] = [];
    if (m.data_type) dataParts.push(m.data_type);
    if (m.data_shape?.length) dataParts.push(`shape=[${m.data_shape.join("×")}]`);
    if (m.n_classes != null) dataParts.push(`${m.n_classes} classes`);
    if (m.n_samples != null) dataParts.push(`${m.n_samples.toLocaleString()} samples`);
    if (dataParts.length) rows.push({ label: "Dataset", value: dataParts.join("  ") });
  }

  if (!rows.length) return <p className="text-slate-400 text-xs">No metadata available.</p>;

  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
      {rows.map(({ label, value }) => (
        <React.Fragment key={label}>
          <dt className="text-slate-400 font-medium whitespace-nowrap">{label}</dt>
          <dd className="font-mono text-slate-700 dark:text-slate-300">{value}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

export default function Summary({ results }: Props) {
  const [openKey, setOpenKey] = useState<string | null>(null);

  const toggleInfo = (key: string) => setOpenKey((prev) => (prev === key ? null : key));

  const downloadCSV = () => {
    const rows = ["model,model_class,tpr_at_0.1pct_fpr,train_accuracy,test_accuracy,dpsgd,risk"];
    results.forEach((m) => {
      const auc = bestAuc(m);
      const tpr = bestTpr(m);
      const { label } = riskLevel(auc);
      rows.push([
        m.model_name,
        m.model_class ?? "",
        tpr?.toFixed(6) ?? "",
        m.train_accuracy?.toFixed(6) ?? "",
        m.test_accuracy?.toFixed(6) ?? "",
        m.dpsgd ? `eps=${m.target_epsilon}` : "No",
        label,
      ].join(","));
    });
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url;
    a.download = "leakpro_summary.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-8">
      {/* Risk cards */}
      {results.length > 1 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {results.map((m) => {
            const auc = bestAuc(m);
            const { label, color, bg } = riskLevel(auc);
            return (
              <div key={`${m.job_id}/${m.model_name}`} className={`rounded-xl p-4 ${bg}`}>
                <p className={`text-2xl font-black ${color}`}>{label}</p>
                <p className="font-bold text-sm mt-1">{m.model_name}</p>
                {m.model_class && <p className="text-xs font-mono text-slate-400 mt-0.5">{m.model_class}</p>}
                <p className="text-xs text-slate-500 mt-1">AUC {auc?.toFixed(3) ?? "—"}</p>
                {m.dpsgd && <p className="text-xs text-blue-500 mt-1">DP-SGD{m.target_epsilon != null ? ` ε=${m.target_epsilon}` : ""}</p>}
              </div>
            );
          })}
        </div>
      )}

      {/* Comparison table */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
        <div className="flex items-center justify-between px-3 py-2 bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Summary</span>
          <button
            onClick={downloadCSV}
            className="flex items-center gap-1 px-2 py-1 rounded border border-slate-300 dark:border-slate-700 text-slate-500 text-xs font-bold hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <span className="material-symbols-outlined text-sm">download</span>
            CSV
          </button>
        </div>
        <table className="w-full text-sm">
          <thead className="bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">Model</th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">TPR@0.1%FPR</th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">Train Acc</th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">Test Acc</th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">DP-SGD</th>
              <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">Risk</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
            {results.map((m) => {
              const key = `${m.job_id}/${m.model_name}`;
              const auc = bestAuc(m);
              const tpr = bestTpr(m);
              const { label, color } = riskLevel(auc);
              const isOpen = openKey === key;
              return (
                <React.Fragment key={key}>
                  <tr className="hover:bg-slate-50/50 dark:hover:bg-slate-900/50 transition-colors">
                    <td className="px-4 py-3 font-semibold">
                      <div className="flex items-center gap-2 flex-wrap">
                        {m.model_name}
                        {m.model_class && (
                          <span className="text-xs px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-500 font-mono">
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
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono">{tpr !== undefined ? (tpr * 100).toFixed(2) + "%" : "—"}</td>
                    <td className="px-4 py-3 font-mono">{pct(m.train_accuracy)}</td>
                    <td className="px-4 py-3 font-mono">{pct(m.test_accuracy)}</td>
                    <td className="px-4 py-3">{m.dpsgd ? <span className="text-blue-500 font-semibold">{m.target_epsilon != null ? `ε=${m.target_epsilon}` : "Yes"}</span> : <span className="text-slate-400">No</span>}</td>
                    <td className="px-4 py-3"><span className={`font-bold ${color}`}>{label}</span></td>
                  </tr>
                  {isOpen && (
                    <tr className="bg-slate-50 dark:bg-slate-900/60">
                      <td colSpan={6} className="px-6 py-4">
                        <MetaPanel r={m} />
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Plain-English verdicts */}
      <div className="flex flex-col gap-3">
        {results.map((m) => {
          const auc = bestAuc(m);
          const { label, color, bg } = riskLevel(auc);
          return (
            <div key={`${m.job_id}/${m.model_name}`} className={`rounded-xl p-5 ${bg}`}>
              <p className={`font-bold mb-1 ${color}`}>
                {m.model_name}
                {m.model_class && <span className="ml-2 text-xs font-mono text-slate-400">{m.model_class}</span>}
                {" "}— {label} risk
              </p>
              <p className="text-sm text-slate-600 dark:text-slate-300">
                {auc !== undefined
                  ? label === "HIGH"
                    ? `AUC = ${auc.toFixed(3)}. An attacker can reliably distinguish training members from non-members.`
                    : label === "MEDIUM"
                      ? `AUC = ${auc.toFixed(3)}. Moderate privacy leakage — some membership signal is detectable.`
                      : `AUC = ${auc.toFixed(3)}. Low privacy leakage — model is reasonably well protected.`
                  : "No attack results available for this model."
                }
                {m.dpsgd && ` Trained with DP-SGD${m.target_epsilon != null ? ` (ε=${m.target_epsilon})` : ""}.`}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
