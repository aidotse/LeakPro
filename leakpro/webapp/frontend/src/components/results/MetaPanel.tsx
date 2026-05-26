import React from "react";
import { ModelResult } from "../../api";

function pct(v: number | undefined) {
  return v !== undefined ? (v * 100).toFixed(1) + "%" : "—";
}

export default function MetaPanel({ r }: { r: ModelResult }) {
  const m = r.train_meta;
  const rows: { label: string; value: string }[] = [];

  if (r.model_class) rows.push({ label: "Architecture", value: r.model_class });

  if (r.dpsgd) {
    const dp = [`DP-SGD  ε=${r.target_epsilon ?? "?"}`];
    if (m?.target_delta != null) dp.push(`δ=${m.target_delta}`);
    if (m?.max_grad_norm != null) dp.push(`clip=${m.max_grad_norm}`);
    if (m?.virtual_batch_size != null) dp.push(`vbs=${m.virtual_batch_size}`);
    rows.push({ label: "Privacy", value: dp.join("  ") });
  } else {
    rows.push({ label: "Privacy", value: "Standard training (no DP)" });
  }

  if (m) {
    if (m.optimizer || m.learning_rate != null || m.batch_size != null) {
      const opt = [m.optimizer ?? "—", `lr=${m.learning_rate ?? "?"}`, `batch=${m.batch_size ?? "?"}`].join("  ");
      rows.push({ label: "Optimizer", value: opt });
    }
    if (m.epochs != null) rows.push({ label: "Epochs", value: String(m.epochs) });
    if (m.f_train != null || m.f_test != null) {
      rows.push({ label: "Data split", value: `train=${pct(m.f_train)}  test=${pct(m.f_test)}` });
    }
    const dataParts: string[] = [];
    if (m.data_type) dataParts.push(m.data_type);
    if (m.data_shape?.length) dataParts.push(`shape=[${m.data_shape.join("×")}]`);
    if (m.n_classes != null) dataParts.push(`${m.n_classes} classes`);
    if (m.n_samples != null) dataParts.push(`${m.n_samples.toLocaleString()} samples`);
    if (dataParts.length) rows.push({ label: "Dataset", value: dataParts.join("  ") });
  }

  if (!rows.length) return <p className="text-slate-400 text-xs">No metadata available.</p>;

  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
      {rows.map(({ label, value }) => (
        <React.Fragment key={label}>
          <dt className="text-slate-400 font-medium whitespace-nowrap">{label}</dt>
          <dd className="font-mono text-slate-700 dark:text-slate-300">{value}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}
