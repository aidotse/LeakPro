"""Base classes and interfaces for modular GIA components.

This module defines all core base classes and data structures used by components.
Everything is in one file for simplicity - no separate interfaces/ subdirectory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from leakpro.fl_utils.fl_client_simulator import ClientObservations


@dataclass
class ComponentMetadata:
    """Metadata describing a component's capabilities and requirements.

    Attributes:
        name: Unique identifier for this component
        display_name: Human-readable name
        description: Detailed description of what this component does
        required_capabilities: Dict of capability_name -> required (True/False)
        paper_reference: Optional citation for the paper this implements

    """

    name: str
    display_name: str = ""
    description: str = ""
    required_capabilities: dict[str, bool] = field(default_factory=dict)
    paper_reference: str = ""

    def __post_init__(self) -> None:
        """Set display_name to name if not provided."""
        if not self.display_name:
            self.display_name = self.name


class Component(ABC):
    """Abstract base class for all modular GIA components.

    All components must implement get_metadata() to declare their
    requirements.
    """

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata describing this component.

        This method must be implemented by all components to declare
        their requirements.
        """
        pass


# =============================================================================
# Label Inference Components
# =============================================================================

@dataclass
class LabelInferenceResult:
    """Result from label inference strategy.

    Attributes:
        labels: Inferred labels (hard or soft)
        confidence: Optional confidence scores per class
        method: Name of the method used
        metadata: Additional method-specific information

    """

    labels: torch.Tensor
    confidence: torch.Tensor | None = None
    method: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


class LabelInferenceStrategy(Component):
    """Base class for label inference strategies.

    Label inference strategies determine or initialize labels from gradients.
    Examples: iDLG (analytical), joint optimization, oracle.
    """

    @abstractmethod
    def infer_labels(
        self,
        gradients: list[torch.Tensor],
        model: nn.Module,
        num_samples: int,
        true_labels: torch.Tensor | None = None,
    ) -> LabelInferenceResult:
        """Infer labels from gradients.

        Args:
            gradients: List of gradient tensors
            model: Target model
            num_samples: Number of samples in the batch
            true_labels: Optional ground-truth labels (for OracleLabels strategy)

        Returns:
            LabelInferenceResult with inferred labels

        """
        pass


# =============================================================================
# Initialization Components
# =============================================================================

@dataclass
class InitializationResult:
    """Result from initialization strategy.

    Attributes:
        reconstruction: Initial reconstruction tensor
        labels: Optional initialized labels
        metadata: Additional initialization information

    """

    reconstruction: torch.Tensor
    labels: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InitializationStrategy(Component):
    """Base class for initialization strategies.

    Initialization strategies create the starting point for reconstruction.
    Examples: random noise, data priors, generator sampling.
    """

    @abstractmethod
    def initialize(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> InitializationResult:
        """Initialize reconstruction.

        Args:
            shape: Shape of reconstruction tensor
            device: Device to create tensor on
            dtype: Data type for tensor

        Returns:
            InitializationResult with initial reconstruction

        """
        pass


# =============================================================================
# Optimization Components
# =============================================================================

@dataclass
class OptimizationState:
    """State maintained during optimization.

    Attributes:
        reconstruction: Current reconstruction
        labels: Current labels (may be optimized, optional)
        iteration: Current iteration number
        loss: Current loss value
        metrics: Dictionary of metric values (includes metadata like loss_history)
        converged: Whether optimization has converged

    """

    reconstruction: torch.Tensor
    labels: torch.Tensor | None = None
    iteration: int = 0
    loss: float = float("inf")
    metrics: dict[str, Any] = field(default_factory=dict)
    converged: bool = False


class OptimizationStrategy(Component):
    """Base class for optimization strategies.

    Optimization strategies define how to iteratively refine the reconstruction.
    Returns final OptimizationState after optimization completes.
    """

    @abstractmethod
    def optimize(
        self,
        reconstruction: torch.Tensor,
        labels: LabelInferenceResult,
        target_model: nn.Module,
        client_observations: ClientObservations,
        proxy_dataloader: DataLoader | None = None,
    ) -> OptimizationState:
        """Run optimization and return final state.

        Args:
            reconstruction: Initial reconstruction tensor
            labels: Label inference result from label inference stage
            target_model: Target model
            client_observations: Observations from client including gradients and BN stats
            proxy_dataloader: Optional proxy data for BN estimation strategies

        Returns:
            Final OptimizationState after optimization

        """
        pass


__all__ = [
    # Core
    "ComponentMetadata",
    "Component",
    # Label Inference
    "LabelInferenceResult",
    "LabelInferenceStrategy",
    # Initialization
    "InitializationResult",
    "InitializationStrategy",
    # Optimization
    "OptimizationState",
    "OptimizationStrategy",
]
