"""Pydantic schemas for LeakPro webapp API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class JobSummary(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    model_count: int = 0


# ---------------------------------------------------------------------------
# Step 2 — Handler / format config
# ---------------------------------------------------------------------------

class DataMeta(BaseModel):
    """Auto-detected metadata from the uploaded dataset."""
    data_type: str          # "image" | "tabular" | "time_series" | "unknown"
    shape: list[int]        # e.g. [3, 32, 32] for images, [n_features] for tabular
    n_samples: int
    n_classes: int | None = None
    dtype: str              # e.g. "float32"
    class_distribution: dict[str, int] | None = None
    label_column: str | None = None


class HandlerConfig(BaseModel):
    """Confirmed handler config (Step 2 output / Step 3 preset choice)."""
    preset: str | None = None      # "cifar_image" | "tabular_csv" | "time_series" | None
    data_type: str
    shape: list[int]
    n_classes: int
    normalise_mean: list[float] | None = None
    normalise_std: list[float] | None = None
    label_column: str | None = None  # for tabular


# ---------------------------------------------------------------------------
# Step 3 — Architecture config
# ---------------------------------------------------------------------------

class ArchConfig(BaseModel):
    """Architecture + training loop source (Step 3 output)."""
    preset: str | None = None      # "cifar_wrn" | "tabular_mlp" | None (custom upload)
    arch_filename: str | None = None
    handler_filename: str | None = None


# ---------------------------------------------------------------------------
# Step 4 — Models
# ---------------------------------------------------------------------------

class TrainParams(BaseModel):
    # `name` becomes a directory name on disk; the pattern is enforced again
    # server-side by security.safe_name.
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    epochs: int = Field(default=50, ge=1, le=1000)
    learning_rate: float = Field(default=0.001, gt=0, le=10)
    batch_size: int = Field(default=128, ge=1, le=16384)
    optimizer: str = "adam"       # "adam" | "sgd"
    f_train: float = Field(default=0.5, gt=0, le=1)   # fraction used for training
    f_test: float = Field(default=0.5, gt=0, le=1)    # fraction used for testing
    dpsgd: bool = False
    target_epsilon: float | None = None
    target_delta: float | None = None
    max_grad_norm: float | None = None
    virtual_batch_size: int | None = None
    accountant: str = "prv"          # "prv" | "rdp"

    @model_validator(mode="after")
    def _fractions_fit(self) -> "TrainParams":
        if self.f_train + self.f_test > 1:
            raise ValueError("f_train + f_test must not exceed 1")
        return self


class ModelInfo(BaseModel):
    """Info card shown after upload or training."""
    name: str
    source: str                  # "uploaded" | "trained"
    input_shape: list[int]
    output_shape: list[int]
    param_count: int
    train_accuracy: float | None = None
    test_accuracy: float | None = None
    learning_rate: float | None = None
    dpsgd: bool = False
    target_epsilon: float | None = None
    target_folder: str = ""


class CompatResult(BaseModel):
    ok: bool
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    param_count: int | None = None
    error: str | None = None
    sample_outputs: list[dict] | None = None
    # Why sample_outputs is missing although the model loaded (logged server-side too).
    sample_error: str | None = None


# ---------------------------------------------------------------------------
# Step 5 — Attack config
# ---------------------------------------------------------------------------

class AttackParams(BaseModel):
    attack: str
    params: dict[str, Any] = {}


class ModelAttackConfig(BaseModel):
    model_name: str
    attacks: list[AttackParams]


# ---------------------------------------------------------------------------
# Step 7 — Results
# ---------------------------------------------------------------------------

class AttackResult(BaseModel):
    attack_name: str
    roc_auc: float | None = None
    tpr_at_fpr: dict[str, float] = {}   # "TPR@10%FPR" etc.
    fpr: list[float] | None = None
    tpr: list[float] | None = None
    signal_values: list[float] | None = None
    true_labels: list[int] | None = None
    # Population index of each entry in signal_values / true_labels, so the UI can
    # fetch the right sample; None when the attack's audit set cannot be mapped.
    audit_indices: list[int] | None = None


class JobListItem(BaseModel):
    job_id: str
    status: str
    created_at: str
    model_names: list[str] = []
    attacks_per_model: dict[str, list[str]] = {}


class ModelResult(BaseModel):
    model_name: str
    source: str
    dpsgd: bool
    target_epsilon: float | None = None
    test_accuracy: float | None = None
    train_accuracy: float | None = None
    model_class: str | None = None   # e.g. "ResNet18_DPsgd"
    job_id: str | None = None        # originating job, used for sample image URLs
    train_meta: dict | None = None   # epochs, lr, batch_size, optimizer, data info
    attacks: list[AttackResult] = []


# ---------------------------------------------------------------------------
# PET optimization — see pet_runner.py for the on-disk layout and
# leakpro/optimization for everything that scores a configuration.
# ---------------------------------------------------------------------------

class PETStartParams(BaseModel):
    """Request to begin (or resume) a privacy-utility search for one model."""

    # Display-side constraint: it dims settings that fall below the user's
    # quality floor, but never gates the search, so moving the slider
    # afterwards re-reads the same results instead of invalidating them.
    max_quality_loss: float = Field(default=0.05, ge=0.0, le=1.0)
    # Any knob named here is pinned to that value; the rest stay searched.
    advanced: dict[str, float] = {}
    n_trials: int | None = Field(default=None, ge=1, le=500)
    # Shadow models per audit during the search; verification always uses more.
    n_shadow: int | None = Field(default=None, ge=1, le=32)
    delta: float | None = Field(default=None, gt=0.0, lt=1.0)
    # Per-sample-gradient memory cap under DP-SGD. Lower it to fit a big model
    # or a busy GPU; it changes speed only, never the privacy accounting.
    max_physical_batch: int = Field(default=32, ge=1, le=4096)
    seed: int = Field(default=0, ge=0)


class PETSetting(BaseModel):
    """One evaluated configuration (an Optuna trial), as the optimization view renders it."""

    index: int
    config: dict[str, float]
    utility: float
    attack_tpr: float | None = None          # None: pruned (see attack_resolution_warning)
    proxy_fpr: float | None = None
    attack_realized_fpr: float | None = None
    attack_resolution_warning: str | None = None
    epsilon: float | None = None             # None: non-private (infinite)
    delta: float | None = None
    accountant: str | None = None
    roc_auc: float | None = None
    tuning_accounted: bool = False


class PETRun(BaseModel):
    """Everything the optimization view needs for one model, read off disk."""

    status: str                               # idle | running | done | failed
    model_name: str
    n_trials: int | None = None
    proxy_fpr: float | None = None
    baseline_utility: float | None = None
    settings: list[PETSetting] = []
    best_indices: list[int] = []
    resolution_warning: str | None = None
    error: str | None = None


class PETVerification(BaseModel):
    """A stronger re-audit of one setting's saved model."""

    status: str                               # running | done | failed
    index: int
    estimated: dict[str, float | None] = {}
    verified: dict[str, Any] | None = None
    error: str | None = None


class PETIndex(BaseModel):
    """Body of the verify / adopt requests."""

    index: int = Field(..., ge=0)
