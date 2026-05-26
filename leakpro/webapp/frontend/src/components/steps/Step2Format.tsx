import React, { useState } from "react";
import { api, DataMeta, HandlerConfig } from "../../api";

interface Props {
  jobId: string;
  meta: DataMeta;
  onDone: (config: HandlerConfig) => void;
}

const TYPE_ICONS: Record<string, string> = {
  image: "image",
  tabular: "table_rows",
  time_series: "show_chart",
  unknown: "help_outline",
};

export default function Step2Format({ jobId, meta, onDone }: Props) {
  const [advanced, setAdvanced] = useState(false);
  const [nClasses, setNClasses] = useState(meta.n_classes ?? 10);
  const [loading, setLoading] = useState(false);

  const confirm = async () => {
    setLoading(true);
    const config: HandlerConfig = {
      data_type: meta.data_type,
      shape: meta.shape,
      n_classes: nClasses,
    };
    await api.setHandlerConfig(jobId, config);
    onDone(config);
    setLoading(false);
  };

  return (
    <div className="flex flex-col gap-8">
      <div className="space-y-2">
        <h2 className="text-4xl font-black tracking-tight">Confirm Format</h2>
        <p className="text-slate-600 dark:text-slate-400 text-lg max-w-2xl">
          We detected the following dataset properties. Review and confirm before continuing.
        </p>
      </div>

      {/* Auto-detected card */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 p-6 bg-slate-50/50 dark:bg-slate-900/50 space-y-4">
        <div className="flex items-center gap-3 mb-2">
          <div className="size-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
            <span className="material-symbols-outlined">{TYPE_ICONS[meta.data_type] ?? "help_outline"}</span>
          </div>
          <div>
            <p className="font-bold capitalize">{meta.data_type.replace("_", " ")} data</p>
            <p className="text-sm text-slate-500">Auto-detected</p>
          </div>
          <span className="ml-auto flex items-center gap-1 text-green-600 dark:text-green-400 text-sm font-bold">
            <span className="material-symbols-outlined text-base">check_circle</span> Detected
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <MetaField label="Shape" value={`[${meta.shape.join(", ")}]`} />
          <MetaField label="Samples" value={meta.n_samples.toLocaleString()} />
          <MetaField label="Classes" value={String(meta.n_classes ?? "—")} />
          <MetaField label="Dtype" value={meta.dtype} />
        </div>

        {/* Advanced override */}
        <button
          onClick={() => setAdvanced(!advanced)}
          className="text-sm text-slate-500 hover:text-primary transition-colors flex items-center gap-1 mt-2"
        >
          <span className="material-symbols-outlined text-base">
            {advanced ? "expand_less" : "expand_more"}
          </span>
          Advanced override
        </button>

        {advanced && (
          <div className="grid grid-cols-2 gap-4 pt-2 border-t border-slate-200 dark:border-slate-700">
            <div>
              <label className="text-xs font-semibold text-slate-500 mb-1 block">Number of classes</label>
              <input
                type="number"
                value={nClasses}
                onChange={(e) => setNClasses(Number(e.target.value))}
                className="w-full rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm px-3 py-2"
              />
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-end">
        <button
          onClick={confirm}
          disabled={loading}
          className="px-8 py-2.5 rounded-lg bg-primary text-white font-bold hover:bg-primary/90 transition-colors flex items-center gap-2 shadow-lg shadow-primary/20 disabled:opacity-50"
        >
          {loading ? "Saving…" : "Confirm & Continue"}
          <span className="material-symbols-outlined text-base">arrow_forward</span>
        </button>
      </div>
    </div>
  );
}

function MetaField({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      <p className="font-mono text-sm font-bold">{value}</p>
    </div>
  );
}
