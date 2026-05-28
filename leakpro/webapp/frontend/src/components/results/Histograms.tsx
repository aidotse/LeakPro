import React, { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import { ModelResult } from "../../api";

interface Props { results: ModelResult[] }

export default function Histograms({ results }: Props) {
  const [modelName, setModelName] = useState(results[0]?.model_name ?? "");
  const model = results.find((m) => m.model_name === modelName);
  const attacks = model?.attacks.filter((a) => a.signal_values && a.true_labels) ?? [];
  const [attackName, setAttackName] = useState(attacks[0]?.attack_name ?? "");

  const attack = attacks.find((a) => a.attack_name === attackName) ?? attacks[0];

  const { members, nonMembers, sigMin, sigMax, sigMed } = useMemo(() => {
    if (!attack?.signal_values || !attack.true_labels) {
      return { members: [], nonMembers: [], sigMin: 0, sigMax: 1, sigMed: 0.5 };
    }
    const sig = attack.signal_values;
    const lbl = attack.true_labels;
    const members = sig.filter((_, i) => lbl[i] === 1);
    const nonMembers = sig.filter((_, i) => lbl[i] === 0);
    const sorted = [...sig].sort((a, b) => a - b);
    const sigMin = sorted[0];
    const sigMax = sorted[sorted.length - 1];
    const sigMed = sorted[Math.floor(sorted.length / 2)];
    return { members, nonMembers, sigMin, sigMax, sigMed };
  }, [attack]);

  const [threshold, setThreshold] = useState<number | null>(null);
  const t = threshold ?? sigMed;

  const metrics = useMemo(() => {
    if (!attack?.signal_values || !attack.true_labels) return null;
    const sig = attack.signal_values;
    const lbl = attack.true_labels;
    const preds = sig.map((s) => (s >= t ? 1 : 0));
    let tp = 0, fp = 0, tn = 0, fn = 0;
    preds.forEach((p, i) => {
      if (p === 1 && lbl[i] === 1) tp++;
      else if (p === 1 && lbl[i] === 0) fp++;
      else if (p === 0 && lbl[i] === 0) tn++;
      else fn++;
    });
    return {
      tpr: tp / (tp + fn) || 0,
      fpr: fp / (fp + tn) || 0,
      acc: (tp + tn) / sig.length,
      tp, fp, tn, fn,
    };
  }, [attack, t]);

  return (
    <div className="flex flex-col gap-6">
      {/* Selectors */}
      <div className="flex gap-4 flex-wrap">
        <Selector label="Model" value={modelName} options={results.map((m) => m.model_name)}
          onChange={(v) => { setModelName(v); setAttackName(""); setThreshold(null); }} />
        <Selector label="Attack" value={attackName} options={attacks.map((a) => a.attack_name)}
          onChange={(v) => { setAttackName(v); setThreshold(null); }} />
      </div>

      {attack ? (
        <>
          <Plot
            data={[
              {
                x: members, type: "histogram", name: "Members", nbinsx: 100,
                marker: { color: "rgba(76, 155, 232, 0.6)" },
              },
              {
                x: nonMembers, type: "histogram", name: "Non-members", nbinsx: 100,
                marker: { color: "rgba(232, 76, 76, 0.6)" },
              },
            ]}
            layout={{
              barmode: "overlay",
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              xaxis: { title: "Attack signal score", gridcolor: "#334155" },
              yaxis: { title: "Count", gridcolor: "#334155" },
              legend: { orientation: "h", y: -0.2 },
              shapes: [{
                type: "line", x0: t, x1: t, y0: 0, y1: 1, yref: "paper",
                line: { color: "#f59e0b", dash: "dash", width: 2 },
              }],
              margin: { t: 20, r: 20, b: 80, l: 60 },
              font: { family: "Inter, sans-serif", color: "#94a3b8" },
            }}
            config={{
              responsive: true, displayModeBar: true, displaylogo: false,
              modeBarButtons: [["toImage"]],
              toImageButtonOptions: { format: "png", filename: "leakpro_histogram", width: 1000, height: 500, scale: 2 },
            }}
            style={{ width: "100%", height: 380 }}
            useResizeHandler
          />

          {/* Threshold slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="font-semibold text-slate-500">Decision threshold</span>
              <span className="font-mono text-amber-500 font-bold">{t.toFixed(4)}</span>
            </div>
            <input
              type="range"
              min={sigMin}
              max={sigMax}
              step={(sigMax - sigMin) / 200}
              value={t}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full accent-amber-500"
            />
          </div>

          {/* Live metrics */}
          {metrics && (
            <div className="grid grid-cols-3 sm:grid-cols-5 gap-3">
              <MetricCard label="TPR" value={(metrics.tpr * 100).toFixed(1) + "%"} />
              <MetricCard label="FPR" value={(metrics.fpr * 100).toFixed(1) + "%"} />
              <MetricCard label="Accuracy" value={(metrics.acc * 100).toFixed(1) + "%"} />
              <MetricCard label="TP / FN" value={`${metrics.tp} / ${metrics.fn}`} />
              <MetricCard label="FP / TN" value={`${metrics.fp} / ${metrics.tn}`} />
            </div>
          )}
        </>
      ) : (
        <p className="text-slate-500">No signal data available for this attack.</p>
      )}
    </div>
  );
}

function Selector({ label, value, options, onChange }: {
  label: string; value: string; options: string[]; onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="text-xs font-semibold text-slate-500 mb-1 block">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm px-3 py-2 min-w-[10rem] pr-8"
      >
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 border border-slate-200 dark:border-slate-800">
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      <p className="font-mono font-bold text-sm">{value}</p>
    </div>
  );
}
