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
  trainModel: (id: string, params: TrainParams) => post(`/jobs/${id}/train`, params),

  // Step 5
  setAttackConfig: (id: string, configs: ModelAttackConfig[]) =>
    post(`/jobs/${id}/attack-config`, configs),

  // Step 6
  startAudit: (id: string) => post(`/jobs/${id}/start`),

  // Step 7
  getResults: (id: string) => get<{ job_id: string; results: ModelResult[] }>(`/jobs/${id}/results`),
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

export interface ModelResult {
  model_name: string;
  source: string;
  dpsgd: boolean;
  target_epsilon?: number;
  test_accuracy?: number;
  attacks: AttackResult[];
}
