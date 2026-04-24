import React, { useMemo, useState } from "react";
import { ModelResult } from "../../api";

// model key is "jobId/modelName" to support cross-session results
function modelKey(m: ModelResult) { return `${m.job_id ?? ""}/${m.model_name}`; }

interface Props { results: ModelResult[]; jobId: string }

export default function Records({ results, jobId }: Props) {
  const [selectedKey, setSelectedKey] = useState(modelKey(results[0]));
  const model = results.find((m) => modelKey(m) === selectedKey) ?? results[0];
  const modelName = model?.model_name ?? "";
  const attacks = model?.attacks.filter((a) => a.signal_values && a.true_labels) ?? [];
  const [attackName, setAttackName] = useState(attacks[0]?.attack_name ?? "");
  const [topN, setTopN] = useState(20);

  const attack = attacks.find((a) => a.attack_name === attackName) ?? attacks[0];

  // Full sorted member list — not sliced, so slider max doesn't depend on topN
  const allMemberRecords = useMemo(() => {
    if (!attack?.signal_values || !attack.true_labels) return [];
    return attack.signal_values
      .map((score, i) => ({ index: i, score, is_member: attack.true_labels![i] }))
      .filter((r) => r.is_member === 1)
      .sort((a, b) => b.score - a.score);
  }, [attack]);

  const topRecords = useMemo(() =>
    allMemberRecords.slice(0, topN).map((r, i) => ({ ...r, rank: i + 1 })),
    [allMemberRecords, topN]
  );

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
        <Selector label="Model" value={selectedKey}
          options={results.map((m) => ({ value: modelKey(m), label: m.model_name }))}
          onChange={(v) => { setSelectedKey(v); setAttackName(""); }} />
        <Selector label="Attack" value={attackName} options={attacks.map((a) => a.attack_name)} onChange={setAttackName} />
        <div>
          <label className="text-xs font-semibold text-slate-500 mb-1 block">Top N records</label>
          <input
            type="range" min={1} max={Math.min(200, allMemberRecords.length || 200)} value={topN}
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
                {["Rank", "Image", "Sample Index", "Risk Score", "Member"].map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {topRecords.map((r) => (
                <tr key={r.rank} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/50">
                  <td className="px-4 py-2.5 font-bold text-slate-400">#{r.rank}</td>
                  <td className="px-4 py-2.5">
                    <img
                      src={`/jobs/${model?.job_id ?? jobId}/sample_image/${r.index}`}
                      alt=""
                      className="w-16 h-16 object-contain rounded border border-slate-200 dark:border-slate-700"
                    />
                  </td>
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
  label: string;
  value: string;
  options: string[] | { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  const normalized = (options as (string | { value: string; label: string })[]).map((o) =>
    typeof o === "string" ? { value: o, label: o } : o
  );
  return (
    <div>
      <label className="text-xs font-semibold text-slate-500 mb-1 block">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm px-3 py-2 min-w-[10rem] pr-8"
      >
        {normalized.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}
