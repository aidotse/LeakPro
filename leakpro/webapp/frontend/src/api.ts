/** Typed fetch wrappers for the LeakPro backend API. */

const BASE = "";
const TOKEN_KEY = "leakpro_api_token";

/**
 * The backend requires a bearer token on every /jobs request. It prints one at
 * startup (or reads LEAKPRO_WEBAPP_TOKEN). We keep it in localStorage and ask
 * for it once per browser.
 */
export function getToken(): string {
  let token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    token = window.prompt("LeakPro API token (printed in the backend log at startup):") ?? "";
    if (token) localStorage.setItem(TOKEN_KEY, token);
  }
  return token;
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

/**
 * Log-stream WebSocket URL for a job. Browsers cannot set an Authorization
 * header on a WebSocket, so the backend takes the token as a query parameter
 * on this one route. Every caller must go through here so an auth change
 * cannot miss a call site again.
 */
export function logsSocketUrl(jobId: string): string {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/jobs/${jobId}/logs?token=${encodeURIComponent(getToken())}`;
}

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  return { Authorization: `Bearer ${getToken()}`, ...(extra ?? {}) };
}

async function handle<T>(res: Response): Promise<T> {
  if (res.status === 401) {
    // Stale or wrong token — drop it so the next call re-prompts.
    clearToken();
    throw new Error("Unauthorized: check the API token printed in the backend log.");
  }
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: authHeaders(body ? { "Content-Type": "application/json" } : undefined),
    body: body ? JSON.stringify(body) : undefined,
  });
  return handle<T>(res);
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path, { headers: authHeaders() });
  return handle<T>(res);
}

async function upload<T>(path: string, file: File, extra?: Record<string, string>): Promise<T> {
  const fd = new FormData();
  fd.append("file", file);
  const url = extra
    ? BASE + path + "?" + new URLSearchParams(extra).toString()
    : BASE + path;
  const res = await fetch(url, { method: "POST", body: fd, headers: authHeaders() });
  return handle<T>(res);
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
    fetch(`/jobs/${id}/models/${encodeURIComponent(modelName)}`, {
      method: "DELETE",
      headers: authHeaders(),
    }).then((r) => handle<unknown>(r)),
  trainModel: (id: string, params: TrainParams) => post(`/jobs/${id}/train`, params),

  // Step 5
  setAttackConfig: (id: string, configs: ModelAttackConfig[]) =>
    post(`/jobs/${id}/attack-config`, configs),

  // Step 6
  startAudit: (id: string) => post(`/jobs/${id}/start`),

  // Step 7
  getResults: (id: string) => get<{ job_id: string; results: ModelResult[] }>(`/jobs/${id}/results`),
  // Risk assessment. All scoring happens in leakpro/risk on the backend; this posts the declared
  // use-case profile and renders whatever comes back.
  assessRisk: (id: string, profile: RiskRequest) => post<RiskResponse>(`/jobs/${id}/risk`, profile),
  getSampleData: (jobId: string, index: number) =>
    get<{ index: number; label: number; features: number[]; feature_names?: string[] }>(`/jobs/${jobId}/sample_data/${index}`),
  // An <img src> cannot carry the Authorization header, so fetch the PNG with
  // the token and hand back an object URL (caller revokes it).
  getSampleImageUrl: async (jobId: string, index: number): Promise<string> => {
    const res = await fetch(`${BASE}/jobs/${jobId}/sample_image/${index}`, { headers: authHeaders() });
    if (res.status === 401) { clearToken(); throw new Error("Unauthorized"); }
    if (!res.ok) throw new Error(await res.text());
    return URL.createObjectURL(await res.blob());
  },

  // PET optimization. Everything that scores a setting runs in leakpro.optimization
  // on the backend; these only start, poll, verify and adopt.
  startOptimization: (id: string, model: string, params: OptimizationParams) =>
    post<{ ok: boolean }>(`/jobs/${id}/pet/start?model_name=${encodeURIComponent(model)}`, params),
  getOptimization: (id: string, model: string) =>
    get<OptimizationRun>(`/jobs/${id}/pet/campaign?model_name=${encodeURIComponent(model)}`),
  verifySetting: (id: string, model: string, index: number, force = false) =>
    post<Verification>(`/jobs/${id}/pet/verify?model_name=${encodeURIComponent(model)}`, { index, force }),
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
  sample_error?: string;
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
  audit_indices?: number[] | null;  // population index per signal entry
}

export interface JobListItem {
  job_id: string;
  status: string;
  created_at: string;
  model_names: string[];
  attacks_per_model?: Record<string, string[]>;
}

export interface TrainMeta {
  /** DP-SGD knobs, recorded on an adopted optimization row. */
  noise_multiplier?: number | null;
  accountant?: string | null;
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
  num_train?: number;
  attacks: AttackResult[];
  /** Set when this row came from an adopted optimization result. */
  optimized?: boolean;
  /** False on an adopted row: the accountant's ε covers one training run; the selection among settings is not accounted. */
  tuning_accounted?: boolean | null;
  /** The adopted setting, its scope and what quality was compared against. */
  pet?: {
    setting_index: number;
    config: SettingConfig;
    scope: VerificationScope;
    proxy_fpr?: number | null;
    attack_tpr?: number | null;
    attack_tpr_ci?: [number, number] | null;
    baseline_utility?: number | null;
    n_shadow?: number;
    notes: string[];
  };
  /** Original backend name, kept when the compare view renames a row for display. */
  orig_model_name?: string;
  /** Measured block per operating point, keyed by String(alpha). Computed by leakpro.risk on the backend. */
  measured?: Record<string, MeasuredAtAlpha>;
}

// ---------------------------------------------------------------------------
// PET optimization (mirrors backend models.py PETSetting / PETRun / PETVerification)
//
// One Setting == one evaluated configuration (an Optuna trial). A pruned trial
// keeps its utility and carries attack_tpr: null with the reason.
// ---------------------------------------------------------------------------

export interface SettingConfig {
  noise_multiplier?: number;
  max_grad_norm?: number;
  learning_rate?: number;
  batch_size?: number;
  [knob: string]: number | undefined;
}

/** Where a measurement was taken: the user's own model, or the search subset. */
export type VerificationScope = "user_model" | "search_subset";

export interface Setting {
  index: number;
  config: SettingConfig;
  /** Task performance on the search subset's held-out split, 0..1. Higher is better. */
  utility: number;
  train_accuracy?: number | null;
  /** Attack success at `proxy_fpr`, 0..1. Lower is safer. null when the trial was pruned. */
  attack_tpr?: number | null;
  /** 95% interval of attack_tpr over the members (audit-point sampling only, a floor on the uncertainty). */
  attack_tpr_ci?: [number, number] | null;
  /** Members the trial trained on (the search subset). */
  n_members?: number | null;
  /** Non-member false alarms the threshold rests on. */
  n_nonmember_false_positives?: number | null;
  /** False-alarm rate the attack figure was measured at (0.01 == 1%). */
  proxy_fpr?: number | null;
  /** The false-alarm rate actually achievable in the data, which can be below
   *  `proxy_fpr` when the score distribution is coarse. */
  attack_realized_fpr?: number | null;
  /** Why attack_tpr is null: the model did not learn, or every score tied. */
  attack_resolution_warning?: string | null;
  /** null means non-private (infinite). */
  epsilon?: number | null;
  delta?: number | null;
  accountant?: string | null;
  roc_auc?: number | null;
  /** Always false today: the search over many configurations is not accounted for. */
  tuning_accounted?: boolean;
}

export interface OptimizationParams {
  /** Fraction of baseline quality the user is willing to give up, 0..1. */
  max_quality_loss: number;
  /** Knobs pinned to a value; omitted keys stay searched. */
  advanced?: Record<string, number>;
  n_trials?: number;
  n_shadow?: number;
  delta?: number;
  max_physical_batch?: number;
  seed?: number;
}

export interface OptimizationRun {
  status: "idle" | "running" | "done" | "failed";
  model_name: string;
  /** Total settings the run intends to test; absent until the run reports it. */
  n_trials?: number | null;
  proxy_fpr?: number | null;
  /** Members per trial: the search subset, not the user's split. */
  n_members?: number | null;
  /** Quality of an unprotected model trained on the same subset: what trial quality is compared against. */
  baseline_utility?: number | null;
  baseline_attack_tpr?: number | null;
  baseline_attack_tpr_ci?: [number, number] | null;
  baseline_source?: "search_subset" | null;
  /** The clicked model's own accuracy, for display; the confirm step compares against it. */
  model_utility?: number | null;
  settings: Setting[];
  /** Indices of the best trade-offs (the frontier). */
  best_indices: number[];
  /** Set when the held-out split could not resolve the default false-alarm rate. */
  resolution_warning?: string | null;
  error?: string | null;
}

export interface Verified {
  /** "user_model": retrained on the model's own split at full size. "search_subset": the search-scale model re-audited. */
  scope: VerificationScope;
  attack_tpr?: number | null;
  attack_tpr_table?: number | null;
  attack_tpr_ci?: [number, number] | null;
  n_members?: number | null;
  n_non_members?: number | null;
  n_nonmember_false_positives?: number | null;
  /** Accuracy on the model's own test split (user_model) or the trial's (search_subset). */
  utility?: number | null;
  train_accuracy?: number | null;
  /** The user's model's own accuracy; null when unknown or not comparable. */
  baseline_utility?: number | null;
  /** ε for one training run at this size; the selection among settings is not accounted. */
  epsilon?: number | null;
  delta?: number | null;
  accountant?: string | null;
  n_shadow?: number;
  proxy_fpr?: number | null;
  realized_fpr?: number | null;
  resolution_warning?: string | null;
  notes: string[];
}

export interface Verification {
  status: "running" | "done" | "failed";
  index: number;
  estimated: {
    attack_tpr?: number | null;
    attack_tpr_ci?: [number, number] | null;
    utility?: number | null;
    n_members?: number | null;
  };
  verified?: Verified | null;
  error?: string | null;
}

// ---------------------------------------------------------------------------
// Risk assessment (mirrors leakpro/risk/schemas.py and backend models.py)
// ---------------------------------------------------------------------------

/** Declared harm parameters. Field names follow Sion et al.'s Loss Magnitude decomposition. */
export interface RiskRequest {
  model_name?: string;
  tolerated_fpr: number;          // alpha; required on purpose, no default
  attacker_prior?: number;        // pi
  n_subjects?: number;            // NDS
  records_per_subject?: number;   // NR
  data_type_sensitivity?: number; // DTS
  subject_type_weight?: number;   // DST
  cost_per_exposed_subject?: number;
  extrapolate_to_population?: boolean;
  notes?: string;
}

export interface RiskMeasured {
  attack_name: string;
  operating_point: number;
  success_rate: number;
  advantage: number;
  lift: number;
  resolvable: boolean;
  roc_auc?: number;
  n_members_audit: number;
  n_non_members_audit: number;
  n_exposed_audit?: number;
  min_resolvable_fpr: number;
}

/** Either the strongest attack's measured block (plus why others were skipped), or why none could be measured. */
export type MeasuredAtAlpha =
  | { measured: RiskMeasured; skipped: string[]; below_chance: string[]; error?: undefined }
  | { error: string; measured?: undefined; skipped?: undefined; below_chance?: undefined };

export interface RiskCombined {
  ppv?: number;
  ppv_balanced?: number;
  gamma: number;
  loss_magnitude?: number;
  loss_event_frequency: number;
  risk?: number;
  expected_exposed_subjects?: number;
  expected_cost?: number;
  extrapolated_exposed_records?: number;
  vulnerability_band?: string;
}

export interface RiskAssessment {
  measured: RiskMeasured;
  declared: Record<string, unknown> & { n_subjects?: number; n_subjects_source: string; attacker_prior: number };
  combined: RiskCombined;
  assumptions: string[];
  warnings: string[];
  policy_version: string;
  leakpro_version: string;
}

export interface RiskResponse {
  job_id: string;
  assessments: Record<string, RiskAssessment>;
  unassessable: Record<string, string>;
}
