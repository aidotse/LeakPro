"""Module that contains the schema definitions for the input handler."""

from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import optuna
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch.nn import Module

ArrayOrScalar = Union[np.ndarray, np.integer, int, list]

class OptimizerConfig(BaseModel):
    """Schema for optimizer parameters."""

    name: str = Field(..., description="Optimizer name")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optimizer parameters")

    @field_validator("name", mode="before")
    @classmethod
    def lowercase_attack_type(cls, v: str) -> str:
        """Convert optimizer name type to lowercase."""
        return v.lower() if isinstance(v, str) else v

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields


class LossConfig(BaseModel):
    """Schema for loss function parameters."""

    name: str = Field(..., description="Loss function name")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Loss function parameters")

    @field_validator("name", mode="before")
    @classmethod
    def lowercase_attack_type(cls, v: str) -> str:
        """Convert loss name type to lowercase."""
        return v.lower() if isinstance(v, str) else v

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields

class DataLoaderConfig(BaseModel):
    """Schema for loss function parameters."""

    params: Dict[str, Any]
    model_config = ConfigDict(extra="forbid")  # Prevent extra fields

class EvalModelConfig(BaseModel):
    """Schema for evaluation model parameters."""

    model_class: str = Field(..., description="Class name of the model")
    module_path: str = Field(..., description="Path to the model module")
    eval_folder: str = Field(..., description="Directory where evaluation model data is stored")

class ReconstructionConfig(BaseModel):
    """Configuration for reconstruction attacks."""

    batch_size: int = Field(32, description="Batch size used during reconstruction")
    num_class_samples: int = Field(1, description="Number of samples to generate for each class")
    num_audited_classes: int = Field(100, description="Number of classes to audit")
    metrics: Dict[str, Any] = Field(default_factory=dict)
    eval_model: Optional[EvalModelConfig] = Field(None, description="Evaluation model configuration")

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields

class AuditConfig(BaseModel):
    """Configuration for the audit process."""

    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    attack_type: Literal["mia", "gia", "minv", "synthetic"] = Field(..., description="Type of attack: must be one of ['mia', 'gia', 'minv', 'synthetic]")  # noqa: E501
    attack_list: List[Dict[str, Any]] = Field(..., min_length=1, description="Must have at least one attack")
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

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields

class TargetConfig(BaseModel):
    """Configuration for the target model."""

    module_path: Annotated[str, Field(pattern=r".*\.py$", description="Path to the target model module")]
    model_class: str = Field(..., description="Class name of the model")
    target_folder: str = Field(..., description="Directory where target model data is stored")
    data_path: str = Field(..., description="Path to dataset file")
    # TODO: Change data_path description to be more descriptive, i.e path to target (or private) dataset.

    # MINV-specific field - optional
    public_data_path: Optional[str] = Field(None, description="Path to the public dataset used for model inversion")

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields

class ShadowModelConfig(BaseModel):
    """Configuration for the Shadow models."""

    model_class: Optional[str] = None
    module_path: Optional[str] = None
    init_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model initialization parameters")
    optimizer: Optional[OptimizerConfig] = Field(..., description="Optimizer configuration")
    criterion: Optional[LossConfig] = Field(..., description="Loss function configuration")
    batch_size: Optional[int] = Field(..., ge=1, description="Batch size used during training")
    epochs: Optional[int] = Field(..., ge=1, description="Number of training epochs")

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields

class DistillationModelConfig(BaseModel):
    """Configuration for the distillation models."""

    pass  # Add fields when necessary


class LeakProConfig(BaseModel):
    """Configuration for the LeakPro framework."""

    audit: AuditConfig
    target: TargetConfig
    shadow_model: Optional[ShadowModelConfig] = Field(None, description="Shadow model config")
    distillation_model: Optional[DistillationModelConfig] = Field(None, description="Distillation model config")
    model_config = ConfigDict(extra="forbid")  # Prevent extra fields


class EvalOutput(BaseModel):
    """Output of the evaluation procedure."""

    accuracy: float = Field(..., ge=0, le=1, description="Accuracy of the model")
    loss: float = Field(..., description="Loss of the model")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional evaluation metrics")

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields


class TrainingOutput(BaseModel):
    """Output of the training procedure."""

    model: Module
    metrics: EvalOutput = Field(..., description="Evaluation metrics")

    # Validate that the model is an instance of torch.nn.Module
    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, v:Module) -> Module:
        """Validate that the model is an instance of torch.nn.Module."""
        if not isinstance(v, Module):
            raise ValueError("model must be an instance of torch.nn.Module")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class MIAMetaDataSchema(BaseModel):
    """Schema for training metadata."""

    train_indices: List[int] = Field(..., min_length=1, description="Indices of training samples")
    test_indices: List[int] = Field(..., min_length=1, description="Indices of testing samples")
    num_train: int = Field(ge=0, description="Number of training samples")
    init_params: Dict[str, Any] = Field(default_factory=dict, description="Model initialization parameters")
    optimizer: OptimizerConfig = Field(..., description="Optimizer configuration")
    criterion: LossConfig = Field(..., description="Loss function configuration")
    data_loader: DataLoaderConfig = Field(..., description="DataLoader configuration")
    epochs: int = Field(ge=1, description="Number of training epochs")
    train_result: EvalOutput = Field(..., description="Final evaluation during training")
    test_result: EvalOutput = Field(..., description="Evaluation output for the test set")
    dataset: str = Field(..., description="Dataset name")

    model_config = ConfigDict(from_attributes=True, extra="forbid")  # Prevent extra fields

class ShadowModelTrainingSchema(BaseModel):
    """Schema for Shadow model metadata storage."""

    init_params: Dict[str, Any] = Field(..., description="Model initialization parameters")
    train_indices: List[int] = Field(..., description="Indices of training samples")
    num_train: int = Field(..., ge=0, description="Number of training samples")
    optimizer: str = Field(..., description="Optimizer name")
    criterion: str = Field(..., description="Criterion (loss function) name")
    epochs: int = Field(..., ge=1, description="Number of training epochs")
    train_result: EvalOutput = Field(..., description="Evaluation output for the training set")
    test_result: EvalOutput = Field(..., description="Evaluation output for the test set")
    online: bool = Field(..., description="Online vs. offline training")
    model_class: str = Field(..., description="Model class name")
    target_model_hash: str = Field(..., description="Hash of target model")

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields

class DistillationModelTrainingSchema(BaseModel):
    """Schema for metadata storage for distillation."""

    init_params: Dict[str, Any] = Field(..., description="Model initialization parameters")
    train_indices: List[int] = Field(..., description="Indices of training samples used for distillation")
    num_train: int = Field(..., ge=0, description="Number of training samples used for distillation")
    optimizer: str = Field(..., description="Optimizer name")
    epochs: int = Field(..., ge=1, description="Number of training epochs")
    label_only: bool = Field(..., description="Whether the distillation process is label-only")

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields

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
    objective: Callable[[Any], float] = Field(lambda x: x, description="Objective function to optimize")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")  # Prevent extra fields

class MIAResultSchema(BaseModel):
    """Schema for the MIA attack results."""

    result_name: str = Field(..., description="Name of the result")
    result_type: str = Field(..., description="Type of the result")
    id: str = Field(..., description="Unique identifier for the result")
    tpr: Union[List[float], None] = Field(..., description="True positive rate")
    fpr: Union[List[float], None] = Field(..., description="False positive rate")
    roc_auc: Union[float, None] = Field(..., description="Area under the ROC curve")
    accuracy: Union[List[float], float] = Field(..., description="Attack accuracy")
    fixed_fpr: Union[Dict[str, float], None] = Field(..., description="Fixed FPR values")
    signal_values: Union[List[float], None] = Field(..., description="Signal values")
    true_labels: List[int] = Field(..., description="True labels")
    config: Dict[str, Any] = Field(..., description="Configuration of the attack")
    tp: Union[ArrayOrScalar, None] = Field(None, description="TP alues")
    fp: Union[ArrayOrScalar, None] = Field(None, description="FP values")
    tn: Union[ArrayOrScalar, None] = Field(None, description="TN values")
    fn: Union[ArrayOrScalar, None] = Field(None, description="FN values")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")  # Prevent extra fields
