#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""RunContext and WorkingState — the two objects that flow through the GIA pipeline.

RunContext: immutable inputs for the duration of an attack run.
WorkingState: mutable state that stages read and update each step.

Stages receive both: ``stage.run(state, ctx, ...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.component_base import LabelInferenceResult
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
        MultiEpochTrainingSimulation,
    )
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.consensus_strategies import (
        AggregationStrategy,
    )
    from leakpro.fl_utils.fl_client_simulator import ClientObservations


@dataclass(frozen=True)
class RunContext:
    """Immutable inputs to an attack run. Built once; shared across all stages.

    Stages may read any field but must not mutate this object.

    Attributes:
        target_model:       The FL-client model being inverted.
        client_observations: Gradients, BN stats, shapes from the FL client.
        training_simulator: Simulates client-side training for gradient recomputation.
        loss_fn:            Task loss (CrossEntropyLoss by default).
        proxy_dataloader:   Auxiliary data for BN-estimation strategies (may be None).
        seed_aggregation:   Strategy for collapsing multi-seed reconstructions.
        epoch_aggregation:  Strategy for collapsing multi-epoch reconstructions (may be None).
        seed:               RNG seed used at attack start for deterministic runs.
    """

    target_model: nn.Module
    client_observations: "ClientObservations"
    training_simulator: "MultiEpochTrainingSimulation"
    loss_fn: nn.Module
    proxy_dataloader: DataLoader | None = None
    seed_aggregation: "AggregationStrategy | None" = None
    epoch_aggregation: "AggregationStrategy | None" = None
    seed: int = 42


@dataclass
class WorkingState:
    """Mutable per-step state. Stages read and update this object.

    Attributes:
        reconstruction:      Data-space tensor being recovered, shape [E, N, G, C, H, W].
        optimizable_tensor:  Tensor actually fed to the optimiser — latent codes when
                             using a GAN/representation, otherwise same as reconstruction.
        labels:              Label inference result (set by bootstrap; may update in joint-label stages).
        metrics:             Free-form bag for per-step scalars, loss history, callbacks.
                             Cross-stage communication uses prefixed keys: ``"stage0/best_seed_idx"``.
        iteration:           Step counter; reset to 0 at the start of each stage.
        loss:                Most recent total loss value.
        converged:           Set True by a stage when it detects early convergence.
                             Set False by the orchestrator when StopStage is raised.
    """

    reconstruction: torch.Tensor | None = None
    optimizable_tensor: torch.Tensor | None = None
    labels: "LabelInferenceResult | None" = None
    metrics: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    loss: float = float("inf")
    converged: bool = False


__all__ = ["RunContext", "WorkingState"]
