/** Typed fetch wrappers for the LeakPro backend API. */

const BASE = "";

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function upload<T>(path: string, file: File, extra?: Record<string, string>): Promise<T> {
  const fd = new FormData();
  fd.append("file", file);
  const url = extra
    ? BASE + path + "?" + new URLSearchParams(extra).toString()
    : BASE + path;
  const res = await fetch(url, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ---------------------------------------------------------------------------

export const api = {
  createJob: () => post<{ job_id: string; status: string; created_at: string }>("/jobs"),
  listJobs: () => get<JobListItem[]>("/jobs"),
  getStatus: (id: string) => get<{ job_id: string; status: string; error?: string }>(`/jobs/${id}/status`),

  // Step 1
  uploadData: (id: string, file: File) => upload<DataMeta>(`/jobs/${id}/upload/data`, file),
  setDataPath: (id: string, path: string) =>
    post<DataMeta>(`/jobs/${id}/data-path`, { path }),

  // Step 2
  setHandlerConfig: (id: string, config: HandlerConfig) =>
    post(`/jobs/${id}/handler-config`, config),

  // Step 1 (dataset handler)
  uploadDatasetHandler: (id: string, file: File) =>
    upload(`/jobs/${id}/upload/dataset-handler`, file),
  setDatasetHandlerPath: (id: string, path: string) =>
    post(`/jobs/${id}/dataset-handler-path`, { path }),

  // Step 3
  uploadArch: (id: string, file: File) => upload(`/jobs/${id}/upload/arch`, file),
  setArchPath: (id: string, path: string) => post(`/jobs/${id}/arch-path`, { path }),
  uploadHandler: (id: string, file: File) => upload(`/jobs/${id}/upload/handler`, file),
  setHandlerPath: (id: string, path: string) => post(`/jobs/${id}/handler-path`, { path }),
  setArchConfig: (id: string, config: ArchConfig) => post(`/jobs/${id}/arch-config`, config),

  // Step 4
  uploadWeights: (id: string, modelName: string, file: File) =>
    upload(`/jobs/${id}/upload/weights`, file, { model_name: modelName }),
  setWeightsPath: (id: string, modelName: string, path: string) =>
    post(`/jobs/${id}/weights-path`, { model_name: modelName, path }),
  uploadModelMetadata: (id: string, modelName: string, file: File) =>
    upload(`/jobs/${id}/upload/model-metadata`, file, { model_name: modelName }),
  setMetadataPath: (id: string, modelName: string, path: string) =>
    post(`/jobs/${id}/model-metadata-path`, { model_name: modelName, path }),
  validateModelMetadata: (id: string, modelName: string) =>
    post<MetaValidationResult>(`/jobs/${id}/validate/model-metadata?model_name=${encodeURIComponent(modelName)}`),
  checkCompat: (id: string, modelName: string) =>
    post<CompatResult>(`/jobs/${id}/check?model_name=${encodeURIComponent(modelName)}`),
  removeModel: (id: string, modelName: string) =>
    fetch(`/jobs/${id}/models/${encodeURIComponent(modelName)}`, { method: "DELETE" }).then((r) => r.json()),
  trainModel: (id: string, params: TrainParams) => post(`/jobs/${id}/train`, params),

  // Step 5
  setAttackConfig: (id: string, configs: ModelAttackConfig[]) =>
    post(`/jobs/${id}/attack-config`, configs),

  // Step 6
  startAudit: (id: string) => post(`/jobs/${id}/start`),

  // Step 7
  getResults: (id: string) => get<{ job_id: string; results: ModelResult[] }>(`/jobs/${id}/results`),
  getSampleData: (jobId: string, index: number) =>
    get<{ index: number; label: number; features: number[]; feature_names?: string[] }>(`/jobs/${jobId}/sample_data/${index}`),

  // PET optimization
  startOptimization: (id: string, model: string, params: OptimizationParams) =>
    post<{ ok: boolean }>(`/jobs/${id}/pet/start?model_name=${encodeURIComponent(model)}`, params),
  getOptimization: (id: string, model: string) =>
    get<OptimizationRun>(`/jobs/${id}/pet/campaign?model_name=${encodeURIComponent(model)}`),
  verifySetting: (id: string, model: string, index: number) =>
    post<Verification>(`/jobs/${id}/pet/verify?model_name=${encodeURIComponent(model)}`, { index }),
  getVerification: (id: string, model: string, index: number) =>
    get<Verification>(`/jobs/${id}/pet/verify?model_name=${encodeURIComponent(model)}&index=${index}`),
  adoptSetting: (id: string, model: string, index: number) =>
    post<{ ok: boolean }>(`/jobs/${id}/pet/adopt?model_name=${encodeURIComponent(model)}`, { index }),
};

// ---------------------------------------------------------------------------
// Types (mirrors backend models.py)
// ---------------------------------------------------------------------------

export interface DataMeta {
  data_type: string;
  shape: number[];
  n_samples: number;
  n_classes?: number;
  dtype: string;
  class_distribution?: Record<string, number>;
  label_column?: string;
}

export interface HandlerConfig {
  preset?: string;
  data_type: string;
  shape: number[];
  n_classes: number;
  normalise_mean?: number[];
  normalise_std?: number[];
  label_column?: string;
}

export interface ArchConfig {
  preset?: string;
  arch_filename?: string;
  handler_filename?: string;
}

export interface CompatResult {
  ok: boolean;
  input_shape?: number[];
  output_shape?: number[];
  param_count?: number;
  error?: string;
  sample_outputs?: Array<{ sample: number; top1_class: number; confidence: number; true_label: number | null }>;
}

export interface MetaValidationResult {
  ok: boolean;
  present_fields: string[];
  missing_fields: string[];
  error?: string;
}

export interface TrainParams {
  name: string;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  optimizer: string;
  f_train: number;
  f_test: number;
  dpsgd: boolean;
  target_epsilon?: number;
  target_delta?: number;
  max_grad_norm?: number;
  virtual_batch_size?: number;
  accountant?: string;
}

export interface AttackParams {
  attack: string;
  params: Record<string, unknown>;
}

export interface ModelAttackConfig {
  model_name: string;
  attacks: AttackParams[];
}

export interface AttackResult {
  attack_name: string;
  roc_auc?: number;
  tpr_at_fpr: Record<string, number>;
  fpr?: number[];
  tpr?: number[];
  signal_values?: number[];
  true_labels?: number[];
}

export interface JobListItem {
  job_id: string;
  status: string;
  created_at: string;
  model_names: string[];
  attacks_per_model?: Record<string, string[]>;
}

export interface TrainMeta {
  epochs?: number;
  learning_rate?: number;
  batch_size?: number;
  optimizer?: string;
  f_train?: number;
  f_test?: number;
  target_delta?: number;
  max_grad_norm?: number;
  virtual_batch_size?: number;
  data_type?: string;
  data_shape?: number[];
  n_classes?: number;
  n_samples?: number;
}

export interface ModelResult {
  model_name: string;
  source: string;
  dpsgd: boolean;
  target_epsilon?: number;
  train_accuracy?: number;
  test_accuracy?: number;
  model_class?: string;
  job_id?: string;
  train_meta?: TrainMeta;
  attacks: AttackResult[];
  /** Set when this row came from an adopted optimization result. */
  optimized?: boolean;
  /** Original backend name, kept when the compare view renames a row for display. */
  orig_model_name?: string;
}

// ---------------------------------------------------------------------------
// PET optimization
//
// One Setting == one evaluated configuration, mirroring a line of the
// campaign's evaluations.jsonl. Field names follow that file so the backend
// can serve records without reshaping them.
// ---------------------------------------------------------------------------

export interface SettingConfig {
  noise_multiplier?: number;
  max_grad_norm?: number;
  learning_rate?: number;
  batch_size?: number;
  [knob: string]: number | undefined;
}

export interface Setting {
  index: number;
  config: SettingConfig;
  /** Task performance, 0..1. Higher is better. */
  utility: number;
  /** Attack success at `proxy_fpr`, 0..1. Lower is safer. Absent if not attacked. */
  attack_tpr?: number;
  attack_tpr_ci95?: [number, number];
  /** False-alarm rate the attack figure was measured at (0.01 == 1%). */
  proxy_fpr?: number;
  /** The false-alarm rate actually achievable in the data, which can be below
   *  `proxy_fpr` when the score distribution is coarse. */
  attack_realized_fpr?: number;
  attack_threshold?: number;
  /** Set when the reported TPR is interpolated rather than observed: the audit
   *  set could not resolve this operating point. Not evidence of privacy. */
  attack_resolution_warning?: string | null;
  /** Distinct nonmember score values — few means a saturated model. */
  attack_distinct_nonmember_scores?: number;
  epsilon?: number;
  delta?: number;
  /** Present when the run skipped the attack for this setting. */
  attack_skipped?: string;
}

export interface OptimizationParams {
  /** Fraction of baseline quality the user is willing to give up, 0..1. */
  max_quality_loss: number;
  /** Optional overrides; omitted keys keep the campaign's own defaults. */
  advanced?: Record<string, number>;
}

export interface OptimizationRun {
  status: "idle" | "running" | "done" | "failed";
  model_name: string;
  /** Total settings the run intends to test; absent until the run reports it. */
  n_configs?: number;
  settings: Setting[];
  /** Indices of the best trade-offs, ascending by utility. */
  best_indices: number[];
  /** Unprotected quality this run is measured against, 0..1. */
  baseline_utility?: number;
  max_quality_loss?: number;
  /** Set when the audit set is too small for the reported figure to mean much. */
  resolution_warning?: string;
  error?: string;
}

export interface Verification {
  status: "running" | "done" | "failed";
  index: number;
  estimated: { attack_tpr: number; utility: number };
  verified?: {
    attack_tpr: number;
    utility: number;
    epsilon?: number;
    realized_fpr?: number;
    resolution_warning?: string | null;
  };
  error?: string;
}
