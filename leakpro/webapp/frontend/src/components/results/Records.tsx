import React, { useMemo, useState, useCallback } from "react";
import { api, ModelResult } from "../../api";

// model key is "jobId/modelName" to support cross-session results
function modelKey(m: ModelResult) { return `${m.job_id ?? ""}/${m.model_name}`; }

interface Props { results: ModelResult[]; jobId: string }

export default function Records({ results, jobId }: Props) {
  const [selectedKey, setSelectedKey] = useState(modelKey(results[0]));
  const model = results.find((m) => modelKey(m) === selectedKey) ?? results[0];
  const modelName = model?.model_name ?? "";
  const attacks = model?.attacks.filter((a) => a.signal_values && a.true_labels) ?? [];
  const [attackName, setAttackName] = useState(attacks[0]?.attack_name ?? "");
  const [topN, setTopN] = useState(5);

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

  const isImage = model?.train_meta?.data_type === "image";
  const headers = isImage
    ? ["Rank", "Image", "Sample Index", "Risk Score", "Member"]
    : ["Rank", "", "Sample Index", "Risk Score", "Member"];

  const [modalData, setModalData] = useState<{ index: number; label: number; features: number[]; feature_names?: string[] } | null>(null);
  const [modalLoading, setModalLoading] = useState(false);

  const openModal = useCallback(async (index: number, jobIdForModel: string) => {
    setModalLoading(true);
    setModalData(null);
    try {
      const d = await api.getSampleData(jobIdForModel, index);
      setModalData(d);
    } finally {
      setModalLoading(false);
    }
  }, []);

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
        <div className="rounded-xl border border-slate-200 dark:border-surface-border overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 dark:bg-surface border-b border-slate-200 dark:border-surface-border">
              <tr>
                {headers.map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-slate-500">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-surface-border">
              {topRecords.map((r) => (
                <React.Fragment key={`${attackName}-${selectedKey}-${r.rank}`}>
                  <tr className="hover:bg-slate-50/50 dark:hover:bg-surface/50">
                    <td className="px-4 py-2.5 font-bold text-slate-400">#{r.rank}</td>
                    {isImage ? (
                      <td className="px-4 py-2.5">
                        <ImgWithLoader src={`/jobs/${model?.job_id ?? jobId}/sample_image/${r.index}`} />
                      </td>
                    ) : (
                      <td className="px-4 py-2.5">
                        <button
                          onClick={() => openModal(r.index, model?.job_id ?? jobId)}
                          className="text-slate-400 hover:text-primary transition-colors"
                          title="View row data"
                        >
                          <span className="material-symbols-outlined text-base">table_rows</span>
                        </button>
                      </td>
                    )}
                    <td className="px-4 py-2.5 font-mono">{r.index}</td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 rounded-full bg-slate-200 dark:bg-surface-2">
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
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-slate-500 text-sm">No records available for this attack.</p>
      )}

      {/* Tabular row modal */}
      {(modalData || modalLoading) && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setModalData(null)}>
          <div className="bg-white dark:bg-surface rounded-xl shadow-2xl w-full max-w-2xl mx-4 overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-surface-border">
              <h3 className="font-bold text-base">
                {modalData ? `Sample #${modalData.index} — Label: ${modalData.label}` : "Loading…"}
              </h3>
              <button onClick={() => setModalData(null)} className="text-slate-400 hover:text-slate-700 dark:hover:text-slate-200">
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            <div className="overflow-y-auto max-h-96 p-6">
              {modalLoading && <p className="text-slate-500 text-sm animate-pulse">Fetching row data…</p>}
              {modalData && (
                <div className="flex flex-col gap-1 font-mono text-xs">
                  {modalData.features.map((v, i) => {
                    const name = modalData.feature_names?.[i] ?? `f${i}`;
                    return (
                      <div key={i} className="flex justify-between gap-4 px-2 py-1 rounded bg-slate-50 dark:bg-surface-2">
                        <span className="text-slate-400 truncate max-w-[70%]" title={name}>{name}</span>
                        <span className="font-semibold shrink-0">{v.toFixed(4)}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ImgWithLoader({ src }: { src: string }) {
  const [loaded, setLoaded] = useState(false);
  return (
    <div className="w-16 h-16 relative">
      {!loaded && (
        <div className="absolute inset-0 rounded border border-slate-200 dark:border-surface-border bg-slate-200 dark:bg-surface-2 animate-pulse" />
      )}
      <img
        src={src}
        alt=""
        className={`w-16 h-16 object-contain rounded border border-slate-200 dark:border-surface-border transition-opacity duration-200 ${loaded ? "opacity-100" : "opacity-0"}`}
        onLoad={() => setLoaded(true)}
        onError={() => setLoaded(true)}
      />
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
        className="rounded border-slate-300 dark:border-surface-border bg-white dark:bg-surface-2 text-sm px-3 py-2 min-w-[10rem] pr-8"
      >
        {normalized.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}
