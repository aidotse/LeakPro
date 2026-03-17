import React, { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import { ModelResult } from "../../api";

const COLOURS = ["#193ce6", "#e61919", "#19e66b", "#e6a819", "#ae19e6"];

interface Props { results: ModelResult[] }

export default function RocChart({ results }: Props) {
  const [selected, setSelected] = useState<string[]>(results.map((m) => m.model_name));

  const traces = useMemo(() => {
    const out: Plotly.Data[] = [
      {
        x: [1e-5, 1], y: [1e-5, 1], mode: "lines",
        line: { dash: "dash", color: "#64748b", width: 1 },
        name: "Random (AUC=0.5)", showlegend: true,
      } as Plotly.Data,
    ];
    results.forEach((model, mi) => {
      if (!selected.includes(model.model_name)) return;
      const col = COLOURS[mi % COLOURS.length];
      model.attacks.forEach((atk) => {
        if (!atk.fpr || !atk.tpr) return;
        const name = `${model.model_name} / ${atk.attack_name} (AUC=${atk.roc_auc?.toFixed(3)})`;
        // Fill area
        out.push({
          x: atk.fpr, y: atk.tpr, mode: "lines", fill: "tozeroy",
          fillcolor: col.replace(")", ", 0.08)").replace("rgb", "rgba"),
          line: { color: "transparent", width: 0 },
          showlegend: false, hoverinfo: "skip",
        } as Plotly.Data);
        // Line
        out.push({
          x: atk.fpr, y: atk.tpr, mode: "lines",
          line: { color: col, width: 2 },
          name, text: atk.fpr.map((f, i) =>
            `FPR: ${f.toFixed(5)}<br>TPR: ${atk.tpr![i].toFixed(5)}`
          ),
          hovertemplate: "%{text}<extra>" + name + "</extra>",
        } as Plotly.Data);
      });
    });
    return out;
  }, [results, selected]);

  return (
    <div className="flex flex-col gap-4">
      {/* Model filter */}
      <div className="flex flex-wrap gap-2">
        {results.map((m, i) => {
          const on = selected.includes(m.model_name);
          return (
            <button
              key={m.model_name}
              onClick={() => setSelected((prev) =>
                on ? prev.filter((n) => n !== m.model_name) : [...prev, m.model_name]
              )}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold border transition-colors
                ${on ? "border-transparent text-white" : "border-slate-300 dark:border-slate-700 text-slate-500"}`}
              style={on ? { backgroundColor: COLOURS[i % COLOURS.length] } : {}}
            >
              {m.model_name}
              {m.dpsgd && " (DP)"}
            </button>
          );
        })}
      </div>

      <Plot
        data={traces}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          xaxis: { type: "log", range: [-5, 0], title: "False Positive Rate", gridcolor: "#334155" },
          yaxis: { type: "log", range: [-5, 0], title: "True Positive Rate", gridcolor: "#334155" },
          legend: { orientation: "h", y: -0.2 },
          margin: { t: 20, r: 20, b: 80, l: 60 },
          font: { family: "Inter, sans-serif", color: "#94a3b8" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: 480 }}
        useResizeHandler
      />

      {/* TPR table */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
        <table className="w-full text-xs">
          <thead className="bg-slate-50 dark:bg-slate-900">
            <tr>
              {["Model", "Attack", "AUC", "TPR@10%", "TPR@1%", "TPR@0.1%", "TPR@0%"].map((h) => (
                <th key={h} className="px-3 py-2 text-left font-bold uppercase tracking-wider text-slate-500">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
            {results.flatMap((m) =>
              m.attacks.map((a) => (
                <tr key={`${m.model_name}-${a.attack_name}`} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/50">
                  <td className="px-3 py-2 font-semibold">{m.model_name}</td>
                  <td className="px-3 py-2">{a.attack_name}</td>
                  <td className="px-3 py-2 font-mono">{a.roc_auc?.toFixed(4) ?? "—"}</td>
                  <td className="px-3 py-2 font-mono">{fmt(a.tpr_at_fpr["TPR@10%FPR"])}</td>
                  <td className="px-3 py-2 font-mono">{fmt(a.tpr_at_fpr["TPR@1%FPR"])}</td>
                  <td className="px-3 py-2 font-mono">{fmt(a.tpr_at_fpr["TPR@0.1%FPR"])}</td>
                  <td className="px-3 py-2 font-mono">{fmt(a.tpr_at_fpr["TPR@0%FPR"])}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function fmt(v?: number) {
  return v !== undefined ? (v * 100).toFixed(2) + "%" : "—";
}
