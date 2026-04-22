import React, { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import { ModelResult } from "../../api";

const COLOURS = [
  "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
  "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
];
const DASHES: Array<"solid"|"dash"|"dot"|"dashdot"|"longdash"> = ["solid","dash","dot","dashdot","longdash"];

type AttackMode = "all" | "best";

interface Props { results: ModelResult[] }

export default function RocChart({ results }: Props) {
  const [selected, setSelected] = useState<string[]>(results.map((m) => m.model_name));
  const [attackMode, setAttackMode] = useState<AttackMode>("all");

  // For each model, determine which attack indices to display
  const visibleAttacks = useMemo(() => {
    const map: Record<string, number[]> = {};
    results.forEach((model) => {
      if (attackMode === "best") {
        // Find the index of the attack with the highest AUC
        let bestIdx = 0;
        let bestAuc = -Infinity;
        model.attacks.forEach((atk, i) => {
          if ((atk.roc_auc ?? -Infinity) > bestAuc) { bestAuc = atk.roc_auc ?? -Infinity; bestIdx = i; }
        });
        map[model.model_name] = [bestIdx];
      } else {
        map[model.model_name] = model.attacks.map((_, i) => i);
      }
    });
    return map;
  }, [results, attackMode]);

  const traces = useMemo(() => {
    const out: Plotly.Data[] = [
      {
        x: [1e-5, 1], y: [1e-5, 1], mode: "lines",
        line: { dash: "dash", color: "#94a3b8", width: 1.5 },
        name: "Random (AUC=0.5)", showlegend: true,
      } as Plotly.Data,
    ];
    results.forEach((model, mi) => {
      if (!selected.includes(model.model_name)) return;
      const col = COLOURS[mi % COLOURS.length];
      const allowedIdx = visibleAttacks[model.model_name] ?? [];
      model.attacks.forEach((atk, ai) => {
        if (!allowedIdx.includes(ai)) return;
        if (!atk.fpr || !atk.tpr) return;
        const name = `${model.model_name} / ${atk.attack_name} (AUC=${atk.roc_auc?.toFixed(3)})`;
        out.push({
          x: atk.fpr, y: atk.tpr, mode: "lines",
          line: { color: col, width: 3, dash: DASHES[ai % DASHES.length] },
          name, text: atk.fpr.map((f, i) =>
            `FPR: ${f.toFixed(5)}<br>TPR: ${atk.tpr![i].toFixed(5)}`
          ),
          hovertemplate: "%{text}<extra>" + name + "</extra>",
        } as Plotly.Data);
      });
    });
    return out;
  }, [results, selected, visibleAttacks]);

  const downloadCSV = () => {
    const rows = ["model,attack,auc,tpr_at_10pct_fpr,tpr_at_1pct_fpr,tpr_at_0.1pct_fpr,tpr_at_0pct_fpr"];
    results.forEach((m) =>
      m.attacks.forEach((a) => {
        rows.push([
          m.model_name, a.attack_name,
          a.roc_auc?.toFixed(6) ?? "",
          (a.tpr_at_fpr["TPR@10%FPR"] ?? ""),
          (a.tpr_at_fpr["TPR@1%FPR"] ?? ""),
          (a.tpr_at_fpr["TPR@0.1%FPR"] ?? ""),
          (a.tpr_at_fpr["TPR@0%FPR"] ?? ""),
        ].join(","));
      })
    );
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url;
    a.download = "leakpro_roc_table.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Model filter */}
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

        {/* Divider */}
        <span className="text-slate-300 dark:text-slate-700">|</span>

        {/* Attack mode filter */}
        {(["all", "best"] as AttackMode[]).map((mode) => (
          <button
            key={mode}
            onClick={() => setAttackMode(mode)}
            className={`px-3 py-1.5 rounded-full text-xs font-bold border transition-colors
              ${attackMode === mode
                ? "bg-slate-700 text-white border-transparent"
                : "border-slate-300 dark:border-slate-700 text-slate-500"}`}
          >
            {mode === "all" ? "All attacks" : "Best per model"}
          </button>
        ))}

        {/* Download CSV */}
        <button
          onClick={downloadCSV}
          className="ml-auto flex items-center gap-1 px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 text-slate-500 text-xs font-bold hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
        >
          <span className="material-symbols-outlined text-sm">download</span>
          CSV
        </button>
      </div>

      <Plot
        data={traces}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          xaxis: { type: "log", range: [-5, 0], title: { text: "False Positive Rate (FPR)" }, gridcolor: "#e2e8f0", gridwidth: 1, zerolinecolor: "#94a3b8" },
          yaxis: { type: "log", range: [-5, 0], title: { text: "True Positive Rate (TPR)" }, gridcolor: "#e2e8f0", gridwidth: 1, zerolinecolor: "#94a3b8" },
          legend: { orientation: "h", y: -0.2 },
          margin: { t: 20, r: 20, b: 80, l: 60 },
          font: { family: "Inter, sans-serif", color: "#94a3b8" },
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtons: [["toImage"]],
          toImageButtonOptions: { format: "png", filename: "leakpro_roc_curves", width: 1200, height: 600, scale: 2 },
        }}
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
