#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Base classes and interfaces for modular GIA components.

This module defines all core base classes and data structures used by components.
Everything is in one file for simplicity - no separate interfaces/ subdirectory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from leakpro.attacks.gia_attacks.modular.core.label_types import (
    BinaryMultilabelType,
    ClassificationLabelType,
    LabelType,
)

import torch
from torch import nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from leakpro.fl_utils.fl_client_simulator import ClientObservations
    from leakpro.attacks.gia_attacks.modular.core.state import RunContext


@dataclass
class ComponentMetadata:
    """Metadata describing a component's capability requirements.

    Attributes:
        name: Unique identifier for this component
        required_capabilities: Dict of capability_name -> required (True/False)
    """

    name: str
    required_capabilities: dict[str, bool] = field(default_factory=dict)


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
        labels:     Inferred labels tensor (no epoch prefix).
        confidence: Optional confidence / probability matrix.
        method:     Name of the inference method used.
        metadata:   Additional method-specific information.
        label_type: Describes the label space and how to manipulate the tensor
                    (e.g. epoch-dim add/strip, confidence conversion).
                    Defaults to :class:`~leakpro.attacks.gia_attacks.modular.
                    core.label_types.ClassificationLabelType` for backward
                    compatibility.

    """

    labels: torch.Tensor
    confidence: torch.Tensor | None = None
    method: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    label_type: LabelType = field(default_factory=ClassificationLabelType)


class LabelInferenceStrategy(Component):
    """Base class for label inference strategies.

    Label inference strategies determine or initialize labels from gradients.
    Examples: iDLG (analytical), joint optimization, oracle.
    """

    @abstractmethod
    def infer(self, ctx: "RunContext") -> LabelInferenceResult:
        """Infer labels from run context.

        Args:
            ctx: Run context with target_model and client_observations
                 (gradients, labels, input_shape, training_settings).

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


class AggregationStrategy(Component):
    """Base class for aggregation strategies.

    Aggregation strategies handle combining multiple reconstruction candidates
    into a consensus reconstruction. This includes:
    - Seed aggregation: Combining multiple random initializations per image
    - Epoch aggregation: Matching and averaging across training epochs (FedAvg)

    When reconstruction has shape [E, N, G, C, H, W] where:
    - E = number of epochs (1 for single-epoch attacks)
    - N = num_images (number of images being reconstructed)
    - G = number of seeds per image
    - C, H, W = image dimensions

    This component computes consensus, used both:
    1. During optimization for group consistency regularization
    2. After optimization for final aggregation

    References:
        - Yin et al., "See through Gradients", CVPR 2021 (seed aggregation)
        - Dimitrov et al., "Data Leakage in Federated Averaging", 2022 (epoch matching)

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
        reconstruction: Current reconstruction in data space [E, N, G, C, H, W]
        optimizable_tensor: Parameters being optimized (may differ from reconstruction)
                           - For pixel-based: same as reconstruction
                           - For latent-based: latent codes [E, N, G, latent_dim]
                           - For feature-based: intermediate features [E, N, G, ...]
        labels: Current labels (may be optimized, optional)
        iteration: Current iteration number
        loss: Current loss value
        metrics: Dictionary of metric values (includes metadata like loss_history)
        converged: Whether optimization has converged

    """

    reconstruction: torch.Tensor
    optimizable_tensor: torch.Tensor | None = None  # None means same as reconstruction
    labels: torch.Tensor | None = None
    iteration: int = 0
    loss: float = float("inf")
    metrics: dict[str, Any] = field(default_factory=dict)
    converged: bool = False


__all__ = [
    "ComponentMetadata",
    "Component",
    "LabelType",
    "ClassificationLabelType",
    "BinaryMultilabelType",
    "LabelInferenceResult",
    "LabelInferenceStrategy",
    "InitializationResult",
    "InitializationStrategy",
    "AggregationStrategy",
    "OptimizationState",
]
