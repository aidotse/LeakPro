import React from "react";
import { ModelResult } from "../../api";

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

export default function Summary({ results }: Props) {
  return (
    <div className="flex flex-col gap-8">
      {/* Risk cards */}
      {results.length > 1 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {results.map((m) => {
            const auc = bestAuc(m);
            const { label, color, bg } = riskLevel(auc);
            return (
              <div key={m.model_name} className={`rounded-xl p-4 ${bg}`}>
                <p className={`text-2xl font-black ${color}`}>{label}</p>
                <p className="font-bold text-sm mt-1">{m.model_name}</p>
                <p className="text-xs text-slate-500 mt-1">AUC {auc?.toFixed(3) ?? "—"}</p>
                {m.dpsgd && <p className="text-xs text-blue-500 mt-1">DP-SGD ε={m.target_epsilon}</p>}
              </div>
            );
          })}
        </div>
      )}

      {/* Comparison table */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800">
            <tr>
              {["Model", "Best AUC", "TPR@0.1%FPR", "Test Acc", "DP-SGD", "Risk"].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
            {results.map((m) => {
              const auc = bestAuc(m);
              const tpr = bestTpr(m);
              const { label, color } = riskLevel(auc);
              return (
                <tr key={m.model_name} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/50 transition-colors">
                  <td className="px-4 py-3 font-semibold">{m.model_name}</td>
                  <td className="px-4 py-3 font-mono">{auc?.toFixed(4) ?? "—"}</td>
                  <td className="px-4 py-3 font-mono">{tpr !== undefined ? (tpr * 100).toFixed(2) + "%" : "—"}</td>
                  <td className="px-4 py-3 font-mono">{m.test_accuracy !== undefined ? (m.test_accuracy * 100).toFixed(1) + "%" : "—"}</td>
                  <td className="px-4 py-3">{m.dpsgd ? <span className="text-blue-500 font-semibold">ε={m.target_epsilon}</span> : <span className="text-slate-400">No</span>}</td>
                  <td className="px-4 py-3"><span className={`font-bold ${color}`}>{label}</span></td>
                </tr>
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
            <div key={m.model_name} className={`rounded-xl p-5 ${bg}`}>
              <p className={`font-bold mb-1 ${color}`}>{m.model_name} — {label} risk</p>
              <p className="text-sm text-slate-600 dark:text-slate-300">
                {auc !== undefined
                  ? label === "HIGH"
                    ? `AUC = ${auc.toFixed(3)}. An attacker can reliably distinguish training members from non-members.`
                    : label === "MEDIUM"
                      ? `AUC = ${auc.toFixed(3)}. Moderate privacy leakage — some membership signal is detectable.`
                      : `AUC = ${auc.toFixed(3)}. Low privacy leakage — model is reasonably well protected.`
                  : "No attack results available for this model."
                }
                {m.dpsgd && ` Trained with DP-SGD (ε=${m.target_epsilon}).`}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
