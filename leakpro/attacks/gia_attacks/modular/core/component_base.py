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
        num_seeds: int = 1,
    ) -> InitializationResult:
        """Initialize reconstruction.

        Args:
            shape: Shape of reconstruction tensor (e.g., [B, C, H, W])
            device: Device to create tensor on
            dtype: Data type for tensor
            num_seeds: Number of seeds per image for multi-seed optimization

        Returns:
            InitializationResult with initial reconstruction.
            Shape will be [B, G, C, H, W] if num_seeds > 1, else [B, C, H, W]

        """
        pass


# =============================================================================
# Seed Aggregation Components
# =============================================================================


class SeedAggregationStrategy(Component):
    """Base class for seed aggregation strategies.

    Seed aggregation handles multiple random seeds per image in multi-seed
    optimization scenarios (e.g., See Through Gradients attack).

    When reconstruction has shape [B, G, C, H, W] where:
    - B = batch size (number of images)
    - G = number of seeds per image
    - C, H, W = image dimensions

    This component computes consensus across seeds, used both:
    1. During optimization for group consistency regularization
    2. After optimization for final aggregation

    Reference:
        Yin et al., "See through Gradients: Image Batch Recovery via
        GradInversion", CVPR 2021
    """

    @abstractmethod
    def compute_consensus(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Compute consensus across seeds.

        Used both during optimization (for group consistency loss) and
        after optimization (for final aggregation).

        Args:
            reconstruction: Tensor of shape [B, G, C, H, W]

        Returns:
            Consensus tensor of shape [B, C, H, W]

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
    # Seed Aggregation
    "SeedAggregationStrategy",
    # Optimization
    "OptimizationState",
    "OptimizationStrategy",
]
