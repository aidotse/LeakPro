import React, { useState } from "react";
import { api, ModelAttackConfig } from "../../api";
import { ModelEntry } from "./Step4Models";

interface Props {
  jobId: string;
  models: ModelEntry[];
  onDone: () => void;
}

// ---------------------------------------------------------------------------
// Attack definitions with configurable params
// ---------------------------------------------------------------------------
interface ParamDef {
  key: string;
  label: string;
  type: "number" | "boolean" | "select";
  default: number | boolean | string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
}

const ATTACK_DEFS: Record<string, { label: string; desc: string; params: ParamDef[] }> = {
  rmia: {
    label: "RMIA",
    desc: "Relative membership inference via likelihood ratio",
    params: [
      { key: "num_shadow_models", label: "Shadow models", type: "number", default: 64, min: 1, max: 256 },
      { key: "temperature",       label: "Temperature",   type: "number", default: 2.0, min: 0.1, max: 10, step: 0.1 },
      { key: "offline",           label: "Offline mode",  type: "boolean", default: false },
    ],
  },
  lira: {
    label: "LiRA",
    desc: "Likelihood ratio attack using shadow models",
    params: [
      { key: "num_shadow_models", label: "Shadow models", type: "number", default: 64, min: 1, max: 256 },
      { key: "online",            label: "Online mode",   type: "boolean", default: true },
    ],
  },
  loss: {
    label: "Loss-based",
    desc: "Threshold on per-sample cross-entropy loss",
    params: [],
  },
  population: {
    label: "Population",
    desc: "Calibrated using population statistics",
    params: [
      { key: "num_population_samples", label: "Population samples", type: "number", default: 1000, min: 100, max: 10000 },
    ],
  },
  gradient: {
    label: "Gradient",
    desc: "Gradient norm as a membership signal",
    params: [],
  },
};

interface AttackInstance {
  id: string;         // unique within model
  attack: string;
  label: string;      // user-editable display name
  params: Record<string, number | boolean | string>;
}

type PerModelInstances = Record<string, AttackInstance[]>;

function makeInstance(attack: string, existingCount: number): AttackInstance {
  const def = ATTACK_DEFS[attack];
  const params: Record<string, number | boolean | string> = {};
  def.params.forEach((p) => { params[p.key] = p.default; });
  return {
    id: `${attack}_${Date.now()}_${existingCount}`,
    attack,
    label: existingCount === 0 ? def.label : `${def.label} (${existingCount + 1})`,
    params,
  };
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function Step5Attacks({ jobId, models, onDone }: Props) {
  const [instances, setInstances] = useState<PerModelInstances>(() =>
    Object.fromEntries(models.map((m) => [m.name, [makeInstance("rmia", 0), makeInstance("loss", 0)]]))
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addInstance = (modelName: string, attack: string) => {
    setInstances((prev) => {
      const existing = (prev[modelName] ?? []).filter((i) => i.attack === attack).length;
      return { ...prev, [modelName]: [...(prev[modelName] ?? []), makeInstance(attack, existing)] };
    });
  };

  const removeInstance = (modelName: string, id: string) =>
    setInstances((prev) => ({ ...prev, [modelName]: prev[modelName].filter((i) => i.id !== id) }));

  const updateInstance = (modelName: string, id: string, patch: Partial<AttackInstance>) =>
    setInstances((prev) => ({
      ...prev,
      [modelName]: prev[modelName].map((i) => i.id === id ? { ...i, ...patch } : i),
    }));

  const updateParam = (modelName: string, id: string, key: string, value: number | boolean | string) =>
    setInstances((prev) => ({
      ...prev,
      [modelName]: prev[modelName].map((i) =>
        i.id === id ? { ...i, params: { ...i.params, [key]: value } } : i
      ),
    }));

  const copyToAll = (sourceName: string) => {
    const source = instances[sourceName] ?? [];
    setInstances(Object.fromEntries(models.map((m) => [m.name, source.map((i) => ({ ...i, id: `${i.attack}_${Date.now()}_${Math.random()}`, params: { ...i.params } }))])));
  };

  const totalInstances = Object.values(instances).reduce((s, arr) => s + arr.length, 0);

  const proceed = async () => {
    setSaving(true); setError(null);
    try {
      const configs: ModelAttackConfig[] = models.map((m) => ({
        model_name: m.name,
        attacks: (instances[m.name] ?? []).map((i) => ({ attack: i.attack, params: { ...i.params, instance_label: i.label } })),
      }));
      await api.setAttackConfig(jobId, configs);
      onDone();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setSaving(false); }
  };

  return (
    <div className="flex flex-col gap-8">
      <div className="space-y-2">
        <h2 className="text-4xl font-black tracking-tight">Configure Attacks</h2>
        <p className="text-slate-600 dark:text-slate-400 text-lg max-w-2xl">
          Add attacks to each model independently. You can add the same attack multiple times
          with different parameters to compare configurations.
        </p>
      </div>

      {models.map((model, mi) => (
        <div key={model.name} className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
          {/* Model header */}
          <div className="flex items-center gap-3 px-6 py-4 bg-slate-50/50 dark:bg-slate-900/50 border-b border-slate-200 dark:border-slate-800">
            <div className="size-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
              <span className="material-symbols-outlined text-sm">psychology</span>
            </div>
            <span className="font-bold">{model.name}</span>
            {model.dpsgd && (
              <span className="text-xs bg-blue-500/10 text-blue-500 px-2 py-0.5 rounded-full font-semibold">
                DP-SGD ε={model.targetEpsilon}
              </span>
            )}
            {mi > 0 && (
              <button onClick={() => copyToAll(models[0].name)}
                className="ml-auto text-xs text-primary hover:underline font-semibold">
                Copy from {models[0].name}
              </button>
            )}
          </div>

          {/* Attack instances */}
          <div className="p-4 flex flex-col gap-3">
            {(instances[model.name] ?? []).length === 0 && (
              <p className="text-sm text-slate-400 italic">No attacks added yet.</p>
            )}
            {(instances[model.name] ?? []).map((inst) => (
              <AttackInstanceCard
                key={inst.id}
                instance={inst}
                onRemove={() => removeInstance(model.name, inst.id)}
                onLabelChange={(l) => updateInstance(model.name, inst.id, { label: l })}
                onParamChange={(k, v) => updateParam(model.name, inst.id, k, v)}
              />
            ))}

            {/* Add attack row */}
            <AddAttackRow onAdd={(attack) => addInstance(model.name, attack)} />
          </div>
        </div>
      ))}

      {error && <p className="text-sm text-red-500">{error}</p>}

      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-500">
          {totalInstances} attack instance{totalInstances !== 1 ? "s" : ""} across {models.length} model{models.length !== 1 ? "s" : ""}
        </p>
        <button onClick={proceed} disabled={totalInstances === 0 || saving}
          className="px-8 py-2.5 rounded-lg bg-primary text-white font-bold hover:bg-primary/90 transition-colors flex items-center gap-2 shadow-lg shadow-primary/20 disabled:opacity-50 disabled:cursor-not-allowed">
          {saving ? "Saving…" : "Continue to Run"}
          <span className="material-symbols-outlined text-base">arrow_forward</span>
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Attack instance card
// ---------------------------------------------------------------------------
function AttackInstanceCard({ instance, onRemove, onLabelChange, onParamChange }: {
  instance: AttackInstance;
  onRemove: () => void;
  onLabelChange: (l: string) => void;
  onParamChange: (k: string, v: number | boolean | string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const def = ATTACK_DEFS[instance.attack];

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
      {/* Header row */}
      <div className="flex items-center gap-3 px-4 py-3 bg-slate-50/50 dark:bg-slate-900/30">
        <span className="text-xs font-bold text-slate-400 bg-slate-200 dark:bg-slate-700 px-2 py-0.5 rounded">
          {def.label}
        </span>
        <input
          value={instance.label}
          onChange={(e) => onLabelChange(e.target.value)}
          className="flex-1 bg-transparent border-b border-dashed border-slate-300 dark:border-slate-600 outline-none text-sm font-semibold focus:border-primary px-0 py-0.5"
          placeholder="Instance name"
        />
        {def.params.length > 0 && (
          <button onClick={() => setExpanded(!expanded)}
            className="text-xs text-slate-500 hover:text-primary flex items-center gap-1 transition-colors">
            <span className="material-symbols-outlined text-sm">tune</span>
            {expanded ? "Hide" : "Params"}
          </button>
        )}
        <button onClick={onRemove} className="text-slate-400 hover:text-red-500 transition-colors">
          <span className="material-symbols-outlined text-base">close</span>
        </button>
      </div>

      {/* Params panel */}
      {expanded && def.params.length > 0 && (
        <div className="px-4 py-3 border-t border-slate-200 dark:border-slate-700 grid grid-cols-2 sm:grid-cols-3 gap-3">
          {def.params.map((p) => (
            <ParamField
              key={p.key}
              def={p}
              value={instance.params[p.key] ?? p.default}
              onChange={(v) => onParamChange(p.key, v)}
            />
          ))}
        </div>
      )}

      {/* Compact param summary when collapsed */}
      {!expanded && def.params.length > 0 && (
        <div className="px-4 py-2 flex gap-3 flex-wrap">
          {def.params.map((p) => (
            <span key={p.key} className="text-xs text-slate-400">
              {p.label}: <span className="font-mono text-slate-600 dark:text-slate-300">{String(instance.params[p.key] ?? p.default)}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Param field
// ---------------------------------------------------------------------------
function ParamField({ def, value, onChange }: {
  def: ParamDef;
  value: number | boolean | string;
  onChange: (v: number | boolean | string) => void;
}) {
  return (
    <div>
      <label className="text-xs font-semibold text-slate-500 mb-1 block">{def.label}</label>
      {def.type === "boolean" ? (
        <label className="flex items-center gap-2 cursor-pointer">
          <div onClick={() => onChange(!value)}
            className={`relative w-9 h-5 rounded-full transition-colors ${value ? "bg-primary" : "bg-slate-300 dark:bg-slate-600"}`}>
            <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${value ? "translate-x-4" : ""}`} />
          </div>
          <span className="text-xs">{value ? "On" : "Off"}</span>
        </label>
      ) : def.type === "select" ? (
        <select value={String(value)} onChange={(e) => onChange(e.target.value)}
          className="w-full rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-xs px-2 py-1.5">
          {def.options?.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input type="number" value={Number(value)}
          min={def.min} max={def.max} step={def.step ?? 1}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-xs px-2 py-1.5 font-mono"
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Add attack dropdown row
// ---------------------------------------------------------------------------
function AddAttackRow({ onAdd }: { onAdd: (attack: string) => void }) {
  const [selected, setSelected] = useState("rmia");
  return (
    <div className="flex items-center gap-2 pt-1">
      <select value={selected} onChange={(e) => setSelected(e.target.value)}
        className="rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm px-3 py-1.5">
        {Object.entries(ATTACK_DEFS).map(([id, def]) => (
          <option key={id} value={id}>{def.label}</option>
        ))}
      </select>
      <button onClick={() => onAdd(selected)}
        className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-primary/10 text-primary font-bold text-sm hover:bg-primary/20 transition-colors">
        <span className="material-symbols-outlined text-base">add</span>
        Add attack
      </button>
    </div>
  );
}
