import React, { useMemo, useState } from "react";
import { ModelResult } from "../../api";

interface Props { results: ModelResult[] }

export default function Records({ results }: Props) {
  const [modelName, setModelName] = useState(results[0]?.model_name ?? "");
  const model = results.find((m) => m.model_name === modelName);
  const attacks = model?.attacks.filter((a) => a.signal_values && a.true_labels) ?? [];
  const [attackName, setAttackName] = useState(attacks[0]?.attack_name ?? "");
  const [topN, setTopN] = useState(20);

  const attack = attacks.find((a) => a.attack_name === attackName) ?? attacks[0];

  const topRecords = useMemo(() => {
    if (!attack?.signal_values || !attack.true_labels) return [];
    const indexed = attack.signal_values
      .map((score, i) => ({ rank: 0, index: i, score, is_member: attack.true_labels![i] }))
      .filter((r) => r.is_member === 1)
      .sort((a, b) => b.score - a.score)
      .slice(0, topN)
      .map((r, i) => ({ ...r, rank: i + 1 }));
    return indexed;
  }, [attack, topN]);

  const downloadCSV = () => {
    const rows = ["rank,audit_index,risk_score,is_member",
      ...topRecords.map((r) => `${r.rank},${r.index},${r.score.toFixed(6)},${r.is_member}`)
    ];
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `leakpro_sensitive_records_${modelName}_${attackName}_top${topN}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Selectors */}
      <div className="flex gap-4 flex-wrap items-end">
        <Selector label="Model" value={modelName} options={results.map((m) => m.model_name)}
          onChange={(v) => { setModelName(v); setAttackName(""); }} />
        <Selector label="Attack" value={attackName} options={attacks.map((a) => a.attack_name)}
          onChange={setAttackName} />
        <div>
          <label className="text-xs font-semibold text-slate-500 mb-1 block">Top N records</label>
          <input
            type="range" min={5} max={Math.min(200, topRecords.length || 200)} value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
            className="w-32 accent-primary"
          />
          <span className="ml-2 text-sm font-mono font-bold">{topN}</span>
        </div>
        <button
          onClick={downloadCSV}
          disabled={topRecords.length === 0}
          className="ml-auto flex items-center gap-2 px-4 py-2 rounded-lg border border-primary text-primary text-sm font-bold hover:bg-primary/5 transition-colors disabled:opacity-40"
        >
          <span className="material-symbols-outlined text-base">download</span>
          Download CSV
        </button>
      </div>

      {topRecords.length > 0 ? (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800">
              <tr>
                {["Rank", "Sample Index", "Risk Score", "Member"].map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {topRecords.map((r) => (
                <tr key={r.rank} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/50">
                  <td className="px-4 py-2.5 font-bold text-slate-400">#{r.rank}</td>
                  <td className="px-4 py-2.5 font-mono">{r.index}</td>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 rounded-full bg-slate-200 dark:bg-slate-700">
                        <div
                          className="h-1.5 rounded-full bg-red-500"
                          style={{ width: `${Math.min(r.score * 100, 100)}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs">{r.score.toFixed(4)}</span>
                    </div>
                  </td>
                  <td className="px-4 py-2.5">
                    <span className="text-xs bg-red-500/10 text-red-500 px-2 py-0.5 rounded-full font-semibold">
                      Member
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-slate-500 text-sm">No records available for this attack.</p>
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
        className="rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm px-3 py-2"
      >
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}
