"""Module that contains the schema definitions for the input handler."""

from typing import Annotated, Any, Callable, Dict, List, Literal, Optional

import optuna
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch.nn import Module

from leakpro.metrics.attack_result import MIAResult


class OptimizerConfig(BaseModel):
    """Schema for optimizer parameters."""

    name: str = Field(..., description="Optimizer name")
    lr: float = Field(default=1e-3, ge=0, description="Learning rate")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay parameter")
    momentum: float = Field(default=0.0, ge=0, description="Momentum parameter")
    dampening: float = Field(default=0.0, ge=0, description="Dampening parameter")
    nesterov: bool = Field(default=False, description="Whether Nesterov momentum is used")

    @field_validator("name", mode="before")
    @classmethod
    def lowercase_attack_type(cls, v: str) -> str:
        """Convert optimizer name type to lowercase."""
        return v.lower() if isinstance(v, str) else v


class LossConfig(BaseModel):
    """Schema for loss function parameters."""

    name: str = Field(..., description="Loss function name")

    @field_validator("name", mode="before")
    @classmethod
    def lowercase_attack_type(cls, v: str) -> str:
        """Convert loss name type to lowercase."""
        return v.lower() if isinstance(v, str) else v

class ReconstructionConfig(BaseModel):
    """Configuration for reconstruction attacks."""

    batch_size: int = Field(32, description="Batch size used during reconstruction")
    num_class_samples: int = Field(1, description="Number of samples to generate for each class")
    num_audited_classes: int = Field(100, description="Number of classes to audit")
    metrics: Dict[str, Any] = Field(default_factory=dict)

class AuditConfig(BaseModel):
    """Configuration for the audit process."""

    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    attack_type: Literal["mia", "gia", "minv", "synthetic"] = Field(..., description="Type of attack: must be one of ['mia', 'gia', 'minv', 'synthetic]")  # noqa: E501
    attack_list: Dict[str, Any] = Field(..., min_length=1, description="Must have at least one attack")
    hyper_param_search: bool = Field(default=False, description="Whether to perform hyperparameter search")
    data_modality: Literal["image", "tabular", "text", "graph", "timeseries"] = Field(..., description="Type of data modality: must be one of ['image', 'tabular', 'text', 'graph', 'timeseries']")  # noqa: E501
    output_dir: str = Field(..., description="Output directory for audit results")

    reconstruction: Optional[ReconstructionConfig] = Field(None, description="Reconstruction attack configuration")

    # turn some of the fields to lowercase
    @field_validator("attack_type", "data_modality", "attack_list", mode="before")
    @classmethod
    def lowercase_attack_type(cls, v: str) -> str:
        """Convert attack type to lowercase."""
        return v.lower() if isinstance(v, str) else v

class TargetConfig(BaseModel):
    """Configuration for the target model."""

    module_path: Annotated[str, Field(pattern=r".*\.py$", description="Path to the target model module")]
    model_class: str = Field(..., description="Class name of the model")
    target_folder: str = Field(..., description="Directory where target model data is stored")
    data_path: str = Field(..., description="Path to dataset file")
    # TODO: Change data_path description to be more descriptive, i.e path to target (or private) dataset.

    # MINV-specific field - optional
    public_data_path: Optional[str] = Field(None, description="Path to the public dataset used for model inversion")

class ShadowModelConfig(BaseModel):
    """Configuration for the Shadow models."""

    model_class: Optional[str] = None
    module_path: Optional[str] = None
    init_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model initialization parameters")
    optimizer: Optional[OptimizerConfig] = Field(..., description="Optimizer configuration")
    loss: Optional[LossConfig] = Field(..., description="Loss function configuration")
    batch_size: Optional[int] = Field(..., ge=1, description="Batch size used during training")
    epochs: Optional[int] = Field(..., ge=1, description="Number of training epochs")

class DistillationModelConfig(BaseModel):
    """Configuration for the distillation models."""

    pass  # Add fields when necessary


class LeakProConfig(BaseModel):
    """Configuration for the LeakPro framework."""

    audit: AuditConfig
    target: TargetConfig
    shadow_model: Optional[ShadowModelConfig] = Field(None, description="Shadow model config")
    distillation_model: Optional[DistillationModelConfig] = Field(None, description="Distillation model config")


class TrainingOutput(BaseModel):
    """Output of the training procedure."""

    model: Module
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # Validate that the model is an instance of torch.nn.Module
    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, v:Module) -> Module:
        """Validate that the model is an instance of torch.nn.Module."""
        if not isinstance(v, Module):
            raise ValueError("model must be an instance of torch.nn.Module")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)

class MIAMetaDataSchema(BaseModel):
    """Schema for training metadata."""

    train_indices: List[int] = Field(..., min_length=1, description="Indices of training samples")
    test_indices: List[int] = Field(..., min_length=1, description="Indices of testing samples")
    num_train: int = Field(ge=0, description="Number of training samples")
    init_params: Dict[str, Any] = Field(default_factory=dict, description="Model initialization parameters")
    optimizer: OptimizerConfig = Field(..., description="Optimizer configuration")
    loss: LossConfig = Field(..., description="Loss function configuration")
    batch_size: int = Field(ge=1, description="Batch size used during training")
    epochs: int = Field(ge=1, description="Number of training epochs")
    train_acc: float = Field(ge=0, le=1, description="Training accuracy (0-1 scale)")
    test_acc: float = Field(ge=0, le=1, description="Test accuracy (0-1 scale)")
    train_loss: float = Field(..., description="Training loss")
    test_loss: float = Field(..., description="Test loss")
    dataset: str = Field(..., description="Dataset name")

    model_config = ConfigDict(from_attributes=True)

class ShadowModelTrainingSchema(BaseModel):
    """Schema for Shadow model metadata storage."""

    init_params: Dict[str, Any] = Field(..., description="Model initialization parameters")
    train_indices: List[int] = Field(..., description="Indices of training samples")
    num_train: int = Field(..., ge=0, description="Number of training samples")
    optimizer: str = Field(..., description="Optimizer name")
    criterion: str = Field(..., description="Criterion (loss function) name")
    batch_size: int = Field(..., ge=1, description="Batch size used during training")
    epochs: int = Field(..., ge=1, description="Number of training epochs")
    train_acc: float = Field(..., ge=0.0, le=1.0, description="Training accuracy (0 to 1)")
    train_loss: float = Field(..., ge=0.0, description="Training loss")
    test_acc: float = Field(..., ge=0.0, le=1.0, description="Test accuracy (0 to 1)")
    test_loss: float = Field(..., ge=0.0, description="Test loss")
    online: bool = Field(..., description="Online vs. offline training")
    model_class: str = Field(..., description="Model class name")
    target_model_hash: str = Field(..., description="Hash of target model")

class DistillationModelTrainingSchema(BaseModel):
    """Schema for metadata storage for distillation."""

    init_params: Dict[str, Any] = Field(..., description="Model initialization parameters")
    train_indices: List[int] = Field(..., description="Indices of training samples used for distillation")
    num_train: int = Field(..., ge=0, description="Number of training samples used for distillation")
    optimizer: str = Field(..., description="Optimizer name")
    batch_size: int = Field(..., ge=1, description="Batch size used during training")
    epochs: int = Field(..., ge=1, description="Number of training epochs")
    label_only: bool = Field(..., description="Whether the distillation process is label-only")

def avg_tpr_at_low_fpr(result: MIAResult) -> float:
    """Calculate the average TPR for FPR values below fpr_threshold.

    This will be used as the default objective function for the hyperparameter search on MIA.
    """
    from numpy import mean
    fpr_threshold = 1e-2
    mask = result.fpr < fpr_threshold
    return float(mean(result.tpr[mask]))

class OptunaConfig(BaseModel):
    """Configuration for the Optuna hyperparameter search."""

    seed: int = Field(default=1234,
                      description="Random seed for reproducibility")
    n_trials: int = Field(default=50,
                          description="Number of trials to find the optimal hyperparameters")
    direction: Literal["maximize", "minimize"] = Field("maximize",
                                                       description="Direction of the optimization, minimize or maximize")
    pruner: optuna.pruners.BasePruner = Field(default=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                              description="Number of steps before pruning of experiments will be available")
    objective: Callable[[MIAResult], float] = Field(default=avg_tpr_at_low_fpr,
                                                    description="Objective function: MIAResult -> float")

    model_config = ConfigDict(arbitrary_types_allowed=True)

