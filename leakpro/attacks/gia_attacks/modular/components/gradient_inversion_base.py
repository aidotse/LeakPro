#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Abstract base class shared by all gradient-inversion optimizers.

This module introduces :class:`GradientInversionBase`, which sits between
:class:`~leakpro.attacks.gia_attacks.modular.core.stage.Stage` and the concrete optimizers
(:class:`~leakpro.attacks.gia_attacks.modular.components.composable_optimizer.ComposableOptimizer`,
:class:`~leakpro.attacks.gia_attacks.modular.components.diffusion_optimizer.DiffusionSamplingOptimizer`,
etc.).

The base class owns the **shared scaffold** that is identical regardless of the
core optimisation loop:

* target gradient extraction from ``client_observations``
* BN component setup via :func:`~leakpro.attacks.gia_attacks.modular.components.
  optimization_building_blocks.optimizer_utils.setup_bn_components`
* dataset normalisation propagation to the representation (if any)
* seed aggregation (post loop)
* epoch aggregation (post loop)
* dimension stripping ``[E, N, G, C, H, W] → [N, C, H, W]``
* final :class:`~leakpro.attacks.gia_attacks.modular.core.component_base.OptimizationState`
  construction

Subclasses only need to implement :meth:`_run_core_loop` with the actual
optimisation logic (Adam/LBFGS iterations, DDPM reverse process, etc.).

"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.optimizer_utils import (
    setup_loss_components,
)
from leakpro.attacks.gia_attacks.modular.core.callbacks import Callback, StopStage
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    AggregationStrategy,
    ComponentMetadata,
    LabelInferenceResult,
    OptimizationState,
)
from leakpro.attacks.gia_attacks.modular.core.stage import Stage
from leakpro.attacks.gia_attacks.modular.core.state import RunContext, WorkingState

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
        LossComponent,
    )

logger = logging.getLogger(__name__)


class GradientInversionBase(Stage):
    """Abstract base for all gradient-inversion optimizers.

    Owns the shared scaffold that is identical regardless of the core loop:

    * target gradient extraction from ``client_observations``
    * BN component setup
    * dataset normalisation propagation to representation
    * seed aggregation (post loop)
    * epoch aggregation (post loop)
    * dimension stripping ``[E, N, G, C, H, W] → [N, C, H, W]``
    * :class:`~leakpro.attacks.gia_attacks.modular.core.component_base.OptimizationState`
      construction

    Subclasses implement only :meth:`_run_core_loop`.

    Args:
        loss_components: List of loss components that compute the
            gradient-inversion objective.
        loss_fn: FL loss function (default: ``CrossEntropyLoss``).
        seed_aggregation: Strategy for aggregating across seeds per image
            after the core loop.  ``None`` = no seed aggregation.
        epoch_aggregation: Strategy for aggregating across epochs after
            the core loop.  ``None`` = no epoch aggregation.
        log_interval: Log progress every *n* steps inside the core loop.
    """

    def __init__(
        self,
        loss_components: List[LossComponent],
        loss_fn: nn.Module | None = None,
        seed_aggregation: AggregationStrategy | None = None,
        epoch_aggregation: AggregationStrategy | None = None,
        log_interval: int = 100,
    ) -> None:
        self.loss_components = loss_components
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.seed_aggregation = seed_aggregation
        self.epoch_aggregation = epoch_aggregation
        self.log_interval = log_interval
        super().__init__()

    # ------------------------------------------------------------------
    # Stage.run — adapter to the optimization scaffold below
    # ------------------------------------------------------------------

    def run(
        self,
        state: WorkingState,
        ctx: RunContext,
        *,
        stage_idx: int = 0,
        callbacks: list[Callback] | None = None,
    ) -> WorkingState:
        """Stage.run adapter: translate (state, ctx) → optimize() args and back.

        ``stage_idx`` and ``callbacks`` are forwarded explicitly through the
        call chain instead of being stored as mutable instance attributes,
        preserving thread-safety and testability.
        """
        cbs: list[Callback] = list(callbacks) if callbacks else []
        try:
            opt_state = self.optimize(
                reconstruction=state.reconstruction,
                labels=state.labels,
                ctx=ctx,
                stage_idx=stage_idx,
                callbacks=cbs,
            )
        except StopStage:
            state.converged = False
            return state

        state.reconstruction = opt_state.reconstruction
        state.optimizable_tensor = opt_state.optimizable_tensor
        if opt_state.labels is not None and state.labels is not None:
            state.labels.labels = opt_state.labels
        state.iteration = opt_state.iteration
        state.loss = opt_state.loss
        state.converged = opt_state.converged
        state.metrics.update(opt_state.metrics)
        return state

    # ------------------------------------------------------------------
    # Callback dispatch helper — used by subclasses inside their core loop
    # ------------------------------------------------------------------

    def _fire_step_callbacks(
        self,
        iteration: int,
        loss_value: float,
        _losses: Any = None,
        *,
        stage_idx: int = 0,
        callbacks: list[Callback] | None = None,
        ctx: RunContext | None = None,
    ) -> None:
        """Fire on_step on every registered callback.

        Called by subclasses from their core loop at ``log_interval`` cadence.
        Raises :exc:`StopStage` if a callback signals early termination.

        Args:
            iteration: Current optimisation step index.
            loss_value: Scalar loss at this step.
            _losses: Optional per-component loss dict (ignored here, may be
                used by future callback implementations).
            stage_idx: Index of the current stage (forwarded from ``run``).
            callbacks: Active callback list (forwarded from ``run``).
            ctx: Immutable run context (forwarded from ``run``).
        """
        if not callbacks:
            return
        snapshot = WorkingState(iteration=iteration, loss=float(loss_value))
        for cb in callbacks:
            cb.on_step(stage_idx, snapshot, ctx)

    # ------------------------------------------------------------------
    # Optimization scaffold
    # ------------------------------------------------------------------

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for this gradient inversion optimizer."""
        return ComponentMetadata(
            name="gradient_inversion_base",
            required_capabilities={"has_gradients": True},
        )

    def optimize(
        self,
        reconstruction: torch.Tensor,
        labels: LabelInferenceResult,
        ctx: RunContext,
        *,
        stage_idx: int = 0,
        callbacks: list[Callback] | None = None,
    ) -> OptimizationState:
        """Shared scaffold. Delegates to :meth:`_run_core_loop`, then aggregates.

        Steps performed by the base class:

        1. Propagate dataset normalisation to representation (if any).
        2. Extract target gradients from ``ctx.client_observations``.
        3. Set up loss components (BN hooks, proxy data, etc.).
        4. Call :meth:`_run_core_loop` — subclass-specific.
        5. Seed aggregation.
        6. Epoch aggregation.
        7. Strip E and G dims: ``[E, N, G, C, H, W] → [N, C, H, W]``.
        8. Strip epoch dim from labels: ``[E, N] → [N]``.
        9. Construct and return the final :class:`OptimizationState`.

        Args:
            reconstruction: Initial reconstruction tensor
                (``[E, N, G, C, H, W]`` or ``[E, N, G, param_dim]`` for
                latent-space optimizers).
            labels: Label inference result from the label inference stage.
            ctx: Immutable run context.
            stage_idx: Index of the current stage (forwarded to callbacks).
            callbacks: Active callback list (forwarded to core loop).

        Returns:
            Final :class:`OptimizationState` with ``reconstruction`` in
            ``[N, C, H, W]``.
        """
        cbs: list[Callback] = list(callbacks) if callbacks else []
        data_mean = ctx.client_observations.data_mean
        data_std = ctx.client_observations.data_std

        # 1. Propagate dataset normalisation to representation
        if hasattr(self, "representation") and self.representation is not None:
            self.representation.prepare_for_stage(data_mean, data_std)

        # 2. Extract target gradients (detached)
        observed_gradients = ctx.client_observations.gradients
        target_grads: List[torch.Tensor] = [
            observed_gradients[name].detach()
            for name, _ in ctx.target_model.named_parameters()
            if name in observed_gradients
        ]

        # 3. Per-component setup (BN hooks, proxy data, etc.)
        setup_loss_components(self.loss_components, reconstruction, ctx)

        # 4. Run the core loop — subclass-specific
        result_state = self._run_core_loop(
            x_init=reconstruction,
            labels=labels,
            target_grads=target_grads,
            ctx=ctx,
            stage_idx=stage_idx,
            callbacks=cbs,
        )
        # result_state.reconstruction must be [E, N, G, C, H, W] (pre-aggregation)

        final_reconstruction = result_state.reconstruction

        # 5. Seed aggregation: [E, N, G, C, H, W] → [E, N, 1, C, H, W]
        if self.seed_aggregation is not None:
            logger.info(
                "  Applying seed aggregation: %s",
                self.seed_aggregation.get_metadata().name,
            )
            # Pass per-seed losses to BestSeedAggregation if available
            if hasattr(self.seed_aggregation, 'set_seed_losses'):
                per_seed_losses = result_state.metrics.get('per_seed_losses')
                if per_seed_losses is not None:
                    self.seed_aggregation.set_seed_losses(per_seed_losses)
                    logger.info(
                        "  Provided per-seed losses shape: %s for best seed selection",
                        tuple(per_seed_losses.shape),
                    )
            final_reconstruction = self.seed_aggregation.compute_consensus(final_reconstruction)

        # 6. Epoch aggregation: [E, N, 1, C, H, W] → [1, N, 1, C, H, W]
        if self.epoch_aggregation is not None:
            logger.info(
                "  Applying epoch aggregation: %s",
                self.epoch_aggregation.get_metadata().name,
            )
            final_reconstruction = self.epoch_aggregation.compute_consensus(final_reconstruction)

        # 7. Strip E and G dims: [E, N, G, C, H, W] → [N, C, H, W]
        if final_reconstruction.ndim == 6:
            final_reconstruction = final_reconstruction[0, :, 0]

        # 8. Strip epoch dim from labels using label_type
        #    [E, N]    → [N]    (ClassificationLabelType)
        #    [E, N, K] → [N, K] (BinaryMultilabelType)
        final_labels = result_state.labels
        if final_labels is not None:
            final_labels = labels.label_type.strip_epoch_dim(final_labels)

        return OptimizationState(
            reconstruction=final_reconstruction,
            optimizable_tensor=result_state.optimizable_tensor,
            labels=final_labels,
            loss=result_state.loss,
            iteration=result_state.iteration,
            converged=result_state.converged,
            metrics=result_state.metrics,
        )

    # ------------------------------------------------------------------
    # Abstract hook for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _run_core_loop(
        self,
        x_init: torch.Tensor,
        labels: LabelInferenceResult,
        target_grads: List[torch.Tensor],
        ctx: RunContext,
        *,
        stage_idx: int = 0,
        callbacks: list[Callback] | None = None,
    ) -> OptimizationState:
        """Core attack loop.

        Must return an :class:`OptimizationState` where:

        * ``reconstruction`` is in ``[E, N, G, C, H, W]`` (pre-aggregation).
          The base class :meth:`optimize` handles aggregation and dimension
          stripping.
        * ``optimizable_tensor`` is either ``None`` or already in its final
          output form (the base class uses it verbatim without further
          stripping).
        * ``labels`` is in ``[E, N]`` (base class strips to ``[N]``).

        Args:
            x_init: Initial reconstruction ``[E, N, G, C, H, W]``
                (or ``[E, N, G, param_dim]`` for latent-space optimizers).
            labels: Full :class:`LabelInferenceResult` (preserves
                ``confidence`` attribute needed by
                :class:`~leakpro.attacks.gia_attacks.modular.components.
                optimization_building_blocks.label_strategies.JointLabelOptimizationStrategy`).
            target_grads: Detached target gradients.
            ctx: Immutable run context.
            stage_idx: Index of the current stage (forwarded from ``optimize``).
            callbacks: Active callback list (forwarded from ``optimize``).

        Returns:
            :class:`OptimizationState` with
            ``reconstruction`` in ``[E, N, G, C, H, W]``.
        """
        ...


__all__ = ["GradientInversionBase"]
