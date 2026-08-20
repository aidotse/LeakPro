import { useMemo, useState } from "react";
import { AttackResult, ModelResult } from "../../api";

const COLOURS = [
  "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
  "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
];

const FPR_OPTIONS = [
  { label: "0%",    value: 0 },
  { label: "0.1%",  value: 0.001 },
  { label: "1%",    value: 0.01 },
  { label: "10%",   value: 0.1 },
];

function getMembersAtFPR(atk: AttackResult, targetFPR: number): Set<number> {
  const sv = atk.signal_values!;
  const tl = atk.true_labels!;
  const nonMemberScores = sv
    .map((s, i) => ({ s, i }))
    .filter(({ i }) => tl[i] === 0)
    .map(({ s }) => s)
    .sort((a, b) => b - a);

  let threshold: number;
  if (targetFPR === 0 || nonMemberScores.length === 0) {
    threshold = nonMemberScores.length > 0 ? nonMemberScores[0] + 1e-9 : Infinity;
  } else {
    const idx = Math.floor(targetFPR * nonMemberScores.length);
    threshold = nonMemberScores[Math.min(idx, nonMemberScores.length - 1)];
  }

  const result = new Set<number>();
  sv.forEach((s, i) => {
    if (tl[i] === 1 && s > threshold) result.add(i);
  });
  return result;
}

function intersect(a: Set<number>, b: Set<number>): Set<number> {
  return new Set([...a].filter((x) => b.has(x)));
}

function union(a: Set<number>, b: Set<number>): Set<number> {
  return new Set([...a, ...b]);
}

function difference(a: Set<number>, b: Set<number>): Set<number> {
  return new Set([...a].filter((x) => !b.has(x)));
}

// ---------------------------------------------------------------------------
// 2-circle SVG Venn (proportional radii)
// ---------------------------------------------------------------------------
function Venn2({
  sets, names, colors,
}: { sets: [Set<number>, Set<number>]; names: [string, string]; colors: [string, string] }) {
  const [A, B] = sets;
  const Aonly = difference(A, B).size;
  const Bonly = difference(B, A).size;
  const AB = intersect(A, B).size;
  const total = union(A, B).size;

  const BASE_R = 95;
  const maxCount = Math.max(A.size, B.size, 1);
  const rA = A.size > 0 ? Math.max(Math.sqrt(A.size / maxCount) * BASE_R, 20) : 0;
  const rB = B.size > 0 ? Math.max(Math.sqrt(B.size / maxCount) * BASE_R, 20) : 0;

  // Position circles: fixed centres, but only draw if radius > 0
  const cAx = 155, cBx = 245, cy = 110;

  return (
    <svg viewBox="0 0 400 230" className="w-full max-w-lg mx-auto">
      {rA > 0 && <circle cx={cAx} cy={cy} r={rA} fill={colors[0]} fillOpacity={0.25} stroke={colors[0]} strokeOpacity={0.8} strokeWidth={2} />}
      {rB > 0 && <circle cx={cBx} cy={cy} r={rB} fill={colors[1]} fillOpacity={0.25} stroke={colors[1]} strokeOpacity={0.8} strokeWidth={2} />}

      {/* A only — show if A has members */}
      {A.size > 0 && <>
        <text x={rA > 0 ? cAx - rA * 0.5 : cAx} y={cy - 6} textAnchor="middle" fontSize={22} fontWeight="bold" fill={colors[0]}>{Aonly}</text>
        <text x={rA > 0 ? cAx - rA * 0.5 : cAx} y={cy + 12} textAnchor="middle" fontSize={10} fill={colors[0]} fontWeight="600">{names[0]}</text>
      </>}

      {/* A∩B — only show if there's overlap */}
      {AB > 0 && <>
        <text x="200" y={cy - 6} textAnchor="middle" fontSize={22} fontWeight="bold" fill="#475569">{AB}</text>
        <text x="200" y={cy + 12} textAnchor="middle" fontSize={9} fill="#64748b">overlap</text>
      </>}

      {/* B only — show if B has members */}
      {B.size > 0 && <>
        <text x={rB > 0 ? cBx + rB * 0.5 : cBx} y={cy - 6} textAnchor="middle" fontSize={22} fontWeight="bold" fill={colors[1]}>{Bonly}</text>
        <text x={rB > 0 ? cBx + rB * 0.5 : cBx} y={cy + 12} textAnchor="middle" fontSize={10} fill={colors[1]} fontWeight="600">{names[1]}</text>
      </>}

      {/* Empty set labels */}
      {A.size === 0 && <text x={cAx - 70} y={cy} textAnchor="middle" fontSize={11} fill={colors[0]} fontStyle="italic">{names[0]}: 0</text>}
      {B.size === 0 && <text x={cBx + 70} y={cy} textAnchor="middle" fontSize={11} fill={colors[1]} fontStyle="italic">{names[1]}: 0</text>}

      {/* footer */}
      <text x="200" y="218" textAnchor="middle" fontSize={11} fill="#94a3b8">
        {total} unique identified members
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// 3-circle SVG Venn
// ---------------------------------------------------------------------------
function Venn3({
  sets, names, colors,
}: { sets: [Set<number>, Set<number>, Set<number>]; names: [string, string, string]; colors: [string, string, string] }) {
  const [A, B, C] = sets;
  const AB = intersect(A, B);
  const AC = intersect(A, C);
  const BC = intersect(B, C);
  const ABC = intersect(AB, C);

  const Aonly = difference(difference(A, B), C).size;
  const Bonly = difference(difference(B, A), C).size;
  const Conly = difference(difference(C, A), B).size;
  const ABonly = difference(AB, C).size;
  const AConly = difference(AC, B).size;
  const BConly = difference(BC, A).size;
  const ABCval = ABC.size;
  const total = union(union(A, B), C).size;

  const cx = 200, r = 80;
  const cA = { x: cx - 48, y: 100 };
  const cB = { x: cx + 48, y: 100 };
  const cC = { x: cx, y: 165 };

  return (
    <svg viewBox="0 0 400 265" className="w-full max-w-lg mx-auto">
      <circle cx={cA.x} cy={cA.y} r={r} fill={colors[0]} fillOpacity={0.22} stroke={colors[0]} strokeOpacity={0.8} strokeWidth={2} />
      <circle cx={cB.x} cy={cB.y} r={r} fill={colors[1]} fillOpacity={0.22} stroke={colors[1]} strokeOpacity={0.8} strokeWidth={2} />
      <circle cx={cC.x} cy={cC.y} r={r} fill={colors[2]} fillOpacity={0.22} stroke={colors[2]} strokeOpacity={0.8} strokeWidth={2} />

      {/* A only */}
      <text x={cA.x - 38} y={90} textAnchor="middle" fontSize={18} fontWeight="bold" fill={colors[0]}>{Aonly}</text>
      <text x={cA.x - 38} y={105} textAnchor="middle" fontSize={9} fill={colors[0]} fontWeight="600">{names[0]}</text>

      {/* B only */}
      <text x={cB.x + 38} y={90} textAnchor="middle" fontSize={18} fontWeight="bold" fill={colors[1]}>{Bonly}</text>
      <text x={cB.x + 38} y={105} textAnchor="middle" fontSize={9} fill={colors[1]} fontWeight="600">{names[1]}</text>

      {/* C only */}
      <text x={cC.x} y={228} textAnchor="middle" fontSize={18} fontWeight="bold" fill={colors[2]}>{Conly}</text>
      <text x={cC.x} y={243} textAnchor="middle" fontSize={9} fill={colors[2]} fontWeight="600">{names[2]}</text>

      {/* A∩B */}
      <text x={cx} y={85} textAnchor="middle" fontSize={15} fontWeight="bold" fill="#475569">{ABonly}</text>

      {/* A∩C */}
      <text x={cx - 52} y={163} textAnchor="middle" fontSize={15} fontWeight="bold" fill="#475569">{AConly}</text>

      {/* B∩C */}
      <text x={cx + 52} y={163} textAnchor="middle" fontSize={15} fontWeight="bold" fill="#475569">{BConly}</text>

      {/* A∩B∩C */}
      <text x={cx} y={135} textAnchor="middle" fontSize={16} fontWeight="bold" fill="#1e293b">{ABCval}</text>

      <text x={cx} y={260} textAnchor="middle" fontSize={11} fill="#94a3b8">
        {total} unique identified members
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Pairwise table for >3 attacks
// ---------------------------------------------------------------------------
function PairwiseTable({
  attackSets, names, colors,
}: { attackSets: Set<number>[]; names: string[]; colors: string[] }) {
  const n = names.length;
  const jaccards: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 1;
      const inter = intersect(attackSets[i], attackSets[j]).size;
      const uni = union(attackSets[i], attackSets[j]).size;
      return uni === 0 ? 0 : inter / uni;
    })
  );

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-slate-400 text-center">
        Venn diagram limited to 3 attacks — showing pairwise Jaccard overlap (|A∩B| / |A∪B|)
      </p>
      <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-surface-border">
        <table className="text-xs w-full">
          <thead>
            <tr>
              <th className="px-3 py-2 bg-slate-50 dark:bg-surface" />
              {names.map((name, i) => (
                <th key={i} className="px-3 py-2 bg-slate-50 dark:bg-surface font-bold text-left" style={{ color: colors[i % colors.length] }}>
                  {name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-surface-border">
            {names.map((rowName, i) => (
              <tr key={i} className="hover:bg-slate-50/50 dark:hover:bg-surface/50">
                <td className="px-3 py-2 font-bold" style={{ color: colors[i % colors.length] }}>{rowName}</td>
                {names.map((_, j) => {
                  const v = jaccards[i][j];
                  return (
                    <td key={j} className="px-3 py-2 font-mono text-center"
                      style={{ background: i === j ? colors[i % colors.length] + "22" : `rgba(79,121,167,${v * 0.4})` }}
                    >
                      {i === j ? "—" : (v * 100).toFixed(1) + "%"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
interface Props { results: ModelResult[] }

export default function Venn({ results }: Props) {
  const [selectedModel, setSelectedModel] = useState(results[0]?.model_name ?? "");
  const [fprValue, setFprValue] = useState(0.01);

  const model = results.find((m) => m.model_name === selectedModel) ?? results[0];

  const validAttacks = useMemo(
    () => (model?.attacks ?? []).filter((a) => a.signal_values && a.true_labels),
    [model]
  );

  const totalMembers = useMemo(() => {
    if (!validAttacks.length) return 0;
    const tl = validAttacks[0].true_labels!;
    return tl.filter((v) => v === 1).length;
  }, [validAttacks]);

  const attackSets = useMemo(
    () => validAttacks.map((a) => getMembersAtFPR(a, fprValue)),
    [validAttacks, fprValue]
  );

  const names = validAttacks.map((a) => a.attack_name);
  const colors = validAttacks.map((_, i) => COLOURS[i % COLOURS.length]);

  return (
    <div className="flex flex-col gap-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Model selector */}
        {results.length > 1 && (
          <div>
            <label className="text-xs font-semibold text-slate-500 mb-1 block">Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="rounded border-slate-300 dark:border-surface-border bg-white dark:bg-surface-2 text-sm px-3 py-2 min-w-[10rem] pr-8"
            >
              {results.map((m) => (
                <option key={m.model_name} value={m.model_name}>{m.model_name}</option>
              ))}
            </select>
          </div>
        )}

        {/* FPR selector */}
        <div className="flex flex-col gap-1">
          <span className="text-xs font-semibold text-slate-500">FPR threshold</span>
          <div className="flex gap-1">
            {FPR_OPTIONS.map((opt) => (
              <button
                key={opt.label}
                onClick={() => setFprValue(opt.value)}
                className={`px-3 py-1.5 rounded-full text-xs font-bold border transition-colors
                  ${fprValue === opt.value
                    ? "bg-slate-700 text-cream border-transparent"
                    : "border-slate-300 dark:border-surface-border text-slate-500"}`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Diagram area */}
      {validAttacks.length === 0 ? (
        <p className="text-slate-500 text-sm py-8 text-center">No attacks have signal data available for this model.</p>
      ) : validAttacks.length === 1 ? (
        <p className="text-slate-500 text-sm py-8 text-center">Need at least 2 attacks to show overlap. This model has only 1 attack with signal data.</p>
      ) : validAttacks.length === 2 ? (
        <Venn2
          sets={[attackSets[0], attackSets[1]]}
          names={[names[0], names[1]]}
          colors={[colors[0], colors[1]]}
        />
      ) : validAttacks.length === 3 ? (
        <Venn3
          sets={[attackSets[0], attackSets[1], attackSets[2]]}
          names={[names[0], names[1], names[2]]}
          colors={[colors[0], colors[1], colors[2]]}
        />
      ) : (
        <PairwiseTable attackSets={attackSets} names={names} colors={COLOURS} />
      )}

      {/* Per-attack stats */}
      {validAttacks.length > 0 && (
        <div className="rounded-xl border border-slate-200 dark:border-surface-border overflow-hidden">
          <div className="px-3 py-2 bg-slate-50 dark:bg-surface border-b border-slate-200 dark:border-surface-border">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
              Members identified at {FPR_OPTIONS.find((o) => o.value === fprValue)?.label} FPR
            </span>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-slate-50 dark:bg-surface border-b border-slate-200 dark:border-surface-border">
              <tr>
                {["Attack", "Identified", "% of Members", "Total Members"].map((h) => (
                  <th key={h} className="px-4 py-2 text-left text-xs font-bold uppercase tracking-wider text-slate-500">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-surface-border">
              {validAttacks.map((atk, i) => {
                const count = attackSets[i].size;
                const pct = totalMembers > 0 ? (count / totalMembers * 100).toFixed(1) : "—";
                return (
                  <tr key={atk.attack_name} className="hover:bg-slate-50/50 dark:hover:bg-surface/50">
                    <td className="px-4 py-2.5 font-semibold flex items-center gap-2">
                      <span className="inline-block w-3 h-3 rounded-full shrink-0" style={{ background: colors[i] }} />
                      {atk.attack_name}
                    </td>
                    <td className="px-4 py-2.5 font-mono">{count}</td>
                    <td className="px-4 py-2.5 font-mono">{totalMembers > 0 ? pct + "%" : "—"}</td>
                    <td className="px-4 py-2.5 font-mono text-slate-400">{totalMembers}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
