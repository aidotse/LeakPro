"""Pydantic schemas for LeakPro webapp API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


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
    name: str
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 128
    optimizer: str = "adam"       # "adam" | "sgd"
    f_train: float = 0.5          # fraction of dataset used for training
    f_test: float = 0.5           # fraction of dataset used for testing
    dpsgd: bool = False
    target_epsilon: float | None = None
    target_delta: float | None = None
    max_grad_norm: float | None = None
    virtual_batch_size: int | None = None


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


class JobListItem(BaseModel):
    job_id: str
    status: str
    created_at: str
    model_names: list[str] = []


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
