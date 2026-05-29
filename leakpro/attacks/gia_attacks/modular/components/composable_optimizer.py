#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Composable optimization strategy built from reusable building blocks.

This provides a flexible way to create gradient inversion attacks by composing:
- Loss components (gradient matching, TV, BN, etc.)
- Constraints (clipping, denoising, etc.)
- Label strategies (fixed, joint optimization, etc.)
- Step strategies (standard, LBFGS, proximal, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from torch import nn, optim

from leakpro.attacks.gia_attacks.modular.components.gradient_inversion_base import GradientInversionBase
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.constraints import (
    ConstraintStrategy,
    FeatureSpaceConstraint,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.label_strategies import (
    FixedLabels,
    LabelStrategy,
)
if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
        LossComponent,
    )
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.optimizer_utils import (
    compute_loss_components,
    log_progress,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.step_strategies import (
    StandardStepStrategy,
    StepStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    TrainingSimulator,
)
from leakpro.attacks.gia_attacks.modular.components.representation_strategies import (
    RepresentationStrategy,
    UnfrozenGANRepresentation,
)
from leakpro.attacks.gia_attacks.modular.config.registry import build_component
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    AggregationStrategy,
    Component,
    ComponentMetadata,
    LabelInferenceResult,
    OptimizationState,
)
from leakpro.attacks.gia_attacks.modular.core.state import RunContext

logger = logging.getLogger(__name__)


@dataclass
class InternalOptimizerState:
    """Internal state maintained during ComposableOptimizer optimization.

    Fields:
        optimizable_tensor: The tensor being optimized (latent codes if using
                           representation, data otherwise)
        best_optimizable_tensor: Best optimizable tensor found so far
        labels: Current labels
        optimizable_params: Additional optimizable parameters (e.g., soft labels)
    """

    optimizable_tensor: torch.Tensor  # What we optimize (latent or data)
    labels: torch.Tensor
    optimizable_params: List[torch.Tensor]  # Additional params like soft labels
    optimizer: optim.Optimizer
    scheduler: Any | None
    iteration: int
    best_loss: float
    best_optimizable_tensor: torch.Tensor | None  # Best params in optimization space
    best_labels: torch.Tensor | None
    aux_data: Dict[str, Any]  # For storing intermediate results


class ComposableOptimizer(GradientInversionBase):
    """Composable optimization strategy built from reusable building blocks.

    Args:
        loss_components: List of loss components to combine
        constraint: Constraint strategy to apply after each step
        label_strategy: How to handle labels during optimization
        step_strategy: How to execute optimization steps
        representation: Strategy for transforming params to pixels (None = pixel space)
        learning_rate: Base learning rate
        max_iterations: Number of optimization iterations
        optimizer_type: ComponentSpec selecting the optimizer (e.g. ``ComponentSpec(type="optimizer.adam")``); None → Adam
        scheduler_type: ComponentSpec selecting the LR scheduler; None → no scheduler
        patience: Early stopping patience (iterations without improvement)

    """

    def __init__(
        self,
        loss_components: List[LossComponent],
        constraint: ConstraintStrategy | None = None,
        label_strategy: LabelStrategy | None = None,
        step_strategy: StepStrategy | None = None,
        representation: RepresentationStrategy | None = None,
        learning_rate: float = 0.1,
        label_learning_rate: float | None = None,
        max_iterations: int = 300,
        optimizer_type: ComponentSpec | None = None,
        scheduler_type: ComponentSpec | None = None,
        patience: int = 10000,
        log_interval: int = None,
        training_simulator: TrainingSimulator | None = None,
        loss_fn: nn.Module | None = None,
        seed_aggregation: AggregationStrategy | None = None,
        epoch_aggregation: AggregationStrategy | None = None,
        freeze_input: bool = False,
        return_best: bool = True,
        verbose: bool = False,
    ) -> None:
        # Delegate shared fields to GradientInversionBase
        super().__init__(loss_components, loss_fn, seed_aggregation, epoch_aggregation, log_interval)
        self.constraint = constraint  # None = no constraint
        self.label_strategy = label_strategy or FixedLabels()
        self.step_strategy = step_strategy or StandardStepStrategy()
        self.representation = representation
        self.learning_rate = learning_rate
        self.label_learning_rate = label_learning_rate if label_learning_rate is not None else learning_rate
        self.max_iterations = max_iterations
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.patience = patience
        self._training_simulator = training_simulator
        self.freeze_input = freeze_input
        self.return_best = return_best  # If True, return best reconstruction; if False, return last
        self.verbose = verbose

    def _compute_reconstruction(
        self,
        optimizable_tensor: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute data-space reconstruction from optimizable parameters.
        
        Args:
            optimizable_tensor: Tensor being optimized (latent codes or data)
            labels: Optional labels for conditional generation
            
        Returns:
            reconstruction: Always in data space [E, N, G, C, H, W]

        """
        if self.representation is not None:
            # Transform from parameter space (e.g., latent) to data space
            reconstruction = self.representation.forward(optimizable_tensor, labels=labels)
            return reconstruction
        # Already in data space (identity)
        return optimizable_tensor

    @property
    def training_simulator(self) -> TrainingSimulator | None:
        """Get the training simulator used by this optimizer."""
        return self._training_simulator

    def get_metadata(self) -> ComponentMetadata:
        """Return metadata for composable optimizer.

        Aggregates required capabilities from all sub-components:
        - loss_components (including BN strategy requirements)
        - label_strategy
        - step_strategy
        - constraint
        - training_simulator
        """
        # Start with base gradient requirement
        all_requirements = {"has_gradients": True}

        # Aggregate from all loss components
        for loss_comp in self.loss_components:
            comp_reqs = self._get_component_requirements(loss_comp)
            all_requirements.update(comp_reqs)

        # Aggregate from other components
        for component in [self.label_strategy, self.step_strategy, self.constraint]:
            if component is not None:
                comp_reqs = self._get_component_requirements(component)
                all_requirements.update(comp_reqs)

        # Check training simulator
        if self._training_simulator is not None:
            comp_reqs = self._get_component_requirements(self._training_simulator)
            all_requirements.update(comp_reqs)

        # Check representation strategy
        if self.representation is not None:
            comp_reqs = self._get_component_requirements(self.representation)
            all_requirements.update(comp_reqs)

        return ComponentMetadata(
            name="composable",
            required_capabilities=all_requirements,
        )

    def _get_component_requirements(self, component: Component) -> dict[str, bool]:
        """Extract required capabilities from a component.

        All components expose requirements via get_metadata().required_capabilities.
        Special case: BNStatisticsRegularization delegates to its internal BNStatisticsStrategy
        for runtime requirements via get_strategy_requirements().

        Args:
            component: Any component that may declare requirements

        Returns:
            Dictionary of required capabilities

        """
        # Get base metadata requirements
        metadata = component.get_metadata()
        requirements = metadata.required_capabilities.copy()

        # Special case: BNStatisticsRegularization needs to check its strategy
        from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (  # noqa: PLC0415
            BNStatisticsRegularization,
        )
        if isinstance(component, BNStatisticsRegularization):
            strategy_reqs = component.get_strategy_requirements()
            requirements.update(strategy_reqs)

        return requirements

    def _run_core_loop(
        self,
        x_init: torch.Tensor,
        labels: LabelInferenceResult,
        target_grads: list[torch.Tensor],
        ctx: RunContext,
        *,
        stage_idx: int = 0,
        callbacks: list | None = None,
    ) -> OptimizationState:
        """Core Adam/LBFGS optimization loop.

        Called by :meth:`GradientInversionBase.optimize` after the shared
        scaffold (gradient extraction, BN setup, normalization propagation)
        has been executed.

        Returns an :class:`OptimizationState` with ``reconstruction`` in
        ``[E, N, G, C, H, W]`` (pre-aggregation).  The base class applies
        seed/epoch aggregation and dimension stripping.
        """
        # Setup optimization state
        state = self._setup_optimization(x_init, labels)

        # Use the configured log_interval (default from constructor)
        log_interval = self.log_interval or 100

        # Default loss value in case max_iterations == 0 (e.g. GIAS stage-2 skipped)
        total_loss_value: float = float("nan")

        # ── Attack-boundary fingerprint (what the LeakPro attack actually received) ──
        if self.verbose:
            g0 = target_grads[0] if target_grads else None
            gnorms = [g.norm().item() for g in target_grads] if target_grads else []
            if g0 is not None:
                logger.info(
                    "[debug \u25b6 leakpro]  grads [%d layers]  "
                    "g[0] norm=%.6f mean=%.3e std=%.3e  "
                    "all: min=%.4f max=%.4f mean=%.4f",
                    len(gnorms), g0.norm().item(), g0.mean().item(), g0.std().item(),
                    min(gnorms), max(gnorms), sum(gnorms) / len(gnorms),
                )
            logger.info(
                "[debug \u25b6 leakpro]  x_init shape=%s  mean=%.6f  std=%.6f",
                list(x_init.shape), x_init.mean().item(), x_init.std().item(),
            )
            for lc in self.loss_components:
                logger.info(
                    "[debug \u25b6 leakpro]  loss   %s: weight=%.2e",
                    type(lc).__name__, lc.weight,
                )

        # Main optimization loop
        for iteration in range(self.max_iterations):
            state.iteration = iteration

            # Define closures for step strategy
            def compute_loss_fn() -> tuple[torch.Tensor, dict[str, float]]:
                return self._compute_loss(state, target_grads, ctx)

            def apply_constraints_fn(s: InternalOptimizerState) -> None:
                if self.constraint is None:
                    return
                if self.representation is None or isinstance(self.constraint, FeatureSpaceConstraint):
                    with torch.no_grad():
                        s.optimizable_tensor.data = self.constraint.apply(
                            s.optimizable_tensor.data, ctx
                        )

            # Execute optimization step using strategy
            total_loss_value, losses = \
                self.step_strategy.execute_step(state, compute_loss_fn, apply_constraints_fn)

            # Check for best reconstruction and early stopping
            should_stop = self._check_early_stop(
                state, total_loss_value, losses
            )

            if should_stop:
                break

            if iteration % log_interval == 0 or iteration == self.max_iterations - 1:
                current_lr = state.optimizer.param_groups[0]["lr"] if state.optimizer is not None else 0.0
                self._log_progress(state, total_loss_value, losses, current_lr)
                self._fire_step_callbacks(
                    iteration, total_loss_value, losses,
                    stage_idx=stage_idx, callbacks=callbacks, ctx=ctx,
                )

        # Get best or last optimizable tensor based on return_best setting
        if self.return_best and state.best_optimizable_tensor is not None:
            # Return best reconstruction (lowest loss across all iterations)
            best_optimizable_tensor = state.best_optimizable_tensor
            final_labels = state.best_labels if state.best_labels is not None else labels
            final_loss = state.best_loss
        else:
            # Return last reconstruction (final iteration)
            best_optimizable_tensor = state.optimizable_tensor.detach()
            final_labels = self.label_strategy.get_labels_for_forward(
                state.optimizable_params, state.labels
            )
            final_loss = total_loss_value

        # Transform to data space for final output
        # (If using representation: latent -> data; otherwise: already in data space)
        with torch.no_grad():
            final_reconstruction = self._compute_reconstruction(
                best_optimizable_tensor,
                labels=final_labels,
            )
        # final_reconstruction is [E, N, G, C, H, W] — the base class will apply
        # seed/epoch aggregation and strip to [N, C, H, W].

        # Compute per-seed gradient matching losses for best seed selection.
        #
        # IMPORTANT: when G > 1, best_optimizable_tensor is saved at the iteration
        # where the *average* loss across all seeds was minimum — NOT at each seed's
        # individual best moment.  Scoring seeds at this shared average-minimum
        # snapshot can make the selected seed's result WORSE than what a single-seed
        # run (G=1) would achieve, because the single-seed run returns its personal
        # best state while multi-seed compares seeds at a moment that may not be
        # optimal for any of them individually.
        #
        # The original GIFD paper runs each restart independently and scores the
        # FINAL state of each restart (choose_optimal / _score_trial).  We match
        # that approach: for G > 1 use the final iteration state so that each seed
        # is scored at the end of its own independent optimisation trajectory.
        per_seed_losses = None
        if final_reconstruction.shape[2] > 1:  # G > 1 (multiple seeds)
            # Use final optimisation state — each seed has evolved independently
            # for max_iterations steps; score them on their final result.
            final_state_tensor = state.optimizable_tensor.detach()
            with torch.no_grad():
                final_state_reconstruction = self._compute_reconstruction(
                    final_state_tensor,
                    labels=final_labels,
                )
            # Replace the reconstruction with the final-state version so that
            # BestSeedAggregation picks a seed AND returns its final-state image.
            final_reconstruction = final_state_reconstruction

            from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (  # noqa: PLC0415
                GradientMatchingLoss,
            )
            for loss_comp in self.loss_components:
                if isinstance(loss_comp, GradientMatchingLoss):
                    try:
                        per_seed_losses = loss_comp.compute_per_seed_losses(
                            final_state_reconstruction, final_labels, target_grads, ctx
                        )
                        # Validate shape
                        E, N, G = final_reconstruction.shape[:3]
                        if per_seed_losses.shape != (E, N, G):
                            logger.warning(
                                f"Per-seed losses shape mismatch: expected [{E}, {N}, {G}], "
                                f"got {list(per_seed_losses.shape)}. Setting to None."
                            )
                            per_seed_losses = None
                    except Exception as e:
                        logger.warning(f"Failed to compute per-seed losses: {e}. Best seed selection disabled.")
                        per_seed_losses = None
                    break

        # Best optimizable tensor in parameter space (e.g., latent codes for GAN).
        # Strip E and G dims here since the base class passes this through unchanged.
        # For G > 1 we switched to the final-state tensor above; use that.
        source_tensor = state.optimizable_tensor.detach() if final_reconstruction.shape[2] > 1 else best_optimizable_tensor
        if self.representation is not None:
            final_optimizable = source_tensor[0, :, 0]  # [E, N, G, param_dim] → [N, param_dim]
        else:
            final_optimizable = None

        # Keep final_labels in [E, N] form — the base class will strip to [N].
        # (final_labels is already [1, N] from _setup_optimization's unsqueeze)

        return OptimizationState(
            reconstruction=final_reconstruction,   # [E, N, G, C, H, W] pre-aggregation
            optimizable_tensor=final_optimizable,  # [N, param_dim] or None (already stripped)
            labels=final_labels,                   # [E, N] — base class strips to [N]
            loss=final_loss,
            iteration=state.iteration,
            converged=(state.aux_data.get("stagnant_iterations", 0) < self.patience),
            metrics={
                "best_loss": state.best_loss,
                "per_seed_losses": per_seed_losses,  # [E, N, G] or None
                **state.aux_data.get("best_losses", {}),
            },
        )

    def _setup_optimization(
        self,
        reconstruction: torch.Tensor,
        labels: LabelInferenceResult,
    ) -> InternalOptimizerState:
        """Setup optimization state and parameters."""

        effective_labels, label_params = self.label_strategy.setup(labels)

        # Label strategy setup complete

        # Ensure data is in the correct format for the optimization space.
        # Two cases:
        #   1. Representation (e.g., GAN): input is in parameter space [E, N, G, latent_dim] (4D).
        #      The representation.forward() will map it to data space [E, N, G, C, H, W] for losses.
        #   2. No representation (pixel space): input should be 6D [E, N, G, C, H, W].
        #      If 4D [N, C, H, W] (e.g., from a previous stage transition), reshape to 6D.
        if self.representation is not None:
            # Parameter space: trust the shape as-is (e.g., [E, N, G, latent_dim] for GAN)
            pass
        elif reconstruction.ndim == 4:
            # Pixel space with 4D input: add epoch (E=1) and seed (G=1) dimensions
            reconstruction = reconstruction.unsqueeze(0).unsqueeze(2)  # [N, C, H, W] -> [1, N, 1, C, H, W]

        # Ensure labels have an epoch dimension as the first dim.
        #
        # After the orchestrator's _reshape_labels_for_epochs the labels on
        # `LabelInferenceResult` already carry the epoch prefix (e.g. [1, N]
        # or [E, N] for classification, [1, N, K] for multi-label).
        # `FixedLabels.setup()` returns those tensors as-is, so no further
        # unsqueeze is required for the normal path.
        #
        # The exception is `JointLabelOptimizationStrategy`, which creates a
        # fresh *bare* soft-label tensor [N, C] from `labels.confidence`.
        # That tensor has no epoch prefix and must be unsqueeze'd here.
        #
        # Fallback: if the orchestrator has no epoch_handling_strategy the
        # labels may still be bare; label_type.bare_ndim tells us whether that
        # is the case.
        label_type = labels.label_type
        if label_params:
            # JointLabelOptimization: effective_labels is bare soft labels [N, C]
            effective_labels = effective_labels.unsqueeze(0)  # → [1, N, C]
        elif effective_labels.ndim == label_type.bare_ndim:
            # Labels arrived without epoch dim (no epoch_handling_strategy in orchestrator).
            effective_labels = label_type.add_epoch_dim(effective_labels)

        # Parameters to optimize
        if not reconstruction.is_leaf:
            reconstruction = reconstruction.detach().clone()

        # Initialise feature-space constraints with the initial value of the
        # optimisable tensor.  This must be done *before* requires_grad is set
        # so that the stored initial point is a clean detached copy.
        if isinstance(self.constraint, FeatureSpaceConstraint):
            self.constraint.set_initial_point(reconstruction.detach().clone())

        if self.freeze_input:
            # freeze_input mode (e.g. GIAS stage 2): keep the input tensor fixed
            # and only optimise the representation's internal parameters (generator weights).
            reconstruction.requires_grad = False

            if not isinstance(self.representation, UnfrozenGANRepresentation):
                raise ValueError(
                    "freeze_input=True requires an UnfrozenGANRepresentation so there are "
                    "parameters to actually optimise."
                )
            gen_params = [p for p in self.representation.generator.parameters() if p.requires_grad]
            param_groups = [{"params": gen_params, "lr": self.learning_rate}]
        else:
            reconstruction.requires_grad = True
            # Create optimizer with separate parameter groups for image and labels
            if label_params:
                # Use separate learning rates for image and label parameters
                param_groups = [
                    {"params": [reconstruction], "lr": self.learning_rate},
                    {"params": label_params, "lr": self.label_learning_rate}
                ]
            else:
                # Only image parameters, single learning rate
                param_groups = [{"params": [reconstruction], "lr": self.learning_rate}]

            # When using an UnfrozenGANRepresentation, also optimise the generator's parameters.
            # Without this the generator weights have gradients but are never updated, making
            # the representation behave identically to the frozen GANRepresentation.
            if isinstance(self.representation, UnfrozenGANRepresentation):
                gen_params = [p for p in self.representation.generator.parameters() if p.requires_grad]
                param_groups.append({"params": gen_params, "lr": self.learning_rate})

        # Create optimizer via registry (falls back to Adam when optimizer_type is None)
        if self.optimizer_type is not None:
            optimizer_builder = build_component(self.optimizer_type)
            optimizer = optimizer_builder(param_groups, lr=self.learning_rate)
        else:
            optimizer = optim.Adam(param_groups, lr=self.learning_rate)

        # Create scheduler via registry (None → no scheduler)
        scheduler = None
        if self.scheduler_type is not None:
            scheduler_builder = build_component(self.scheduler_type)
            scheduler = scheduler_builder(optimizer, self.max_iterations)

        return InternalOptimizerState(
            optimizable_tensor=reconstruction,  # Explicit: this is what we optimize
            labels=effective_labels,
            optimizable_params=label_params,
            optimizer=optimizer,
            scheduler=scheduler,
            iteration=0,
            best_loss=float("inf"),
            best_optimizable_tensor=None,
            best_labels=None,
            aux_data={},
        )

    def _compute_loss(
        self,
        state: InternalOptimizerState,
        target_gradients: List[torch.Tensor],
        ctx: RunContext,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss from all components.

        Args:
            state: Current optimization state
            target_gradients: True gradients to match
            ctx: Run context providing target_model and loss_fn

        Returns:
            Total loss and dict of individual component losses

        """
        # Get effective labels for this iteration
        effective_labels = self.label_strategy.get_labels_for_forward(
            state.optimizable_params, state.labels
        )

        # Compute data-space reconstruction from optimizable parameters
        # (If using representation: latent -> data; otherwise: identity)
        reconstruction = self._compute_reconstruction(
            state.optimizable_tensor,
            labels=effective_labels,
        )

        # Inject latent reference into LatentKLDivergenceRegularization components
        # (must happen after reconstruction so gradients flow through z → G(z) → loss)
        from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
            LatentKLDivergenceRegularization,
        )
        for comp in self.loss_components:
            if isinstance(comp, LatentKLDivergenceRegularization):
                comp.set_latent(state.optimizable_tensor)

        # All loss components operate on data-space reconstruction
        return compute_loss_components(
            self.loss_components, reconstruction, effective_labels, target_gradients, ctx
        )

    def _log_progress(
        self,
        state: InternalOptimizerState,
        total_loss: float,
        losses: dict[str, float],
        current_lr: float = 0.0,
    ) -> None:
        """Log optimization progress."""
        suffix = ""
        latent_code = None

        if state.optimizable_params:
            soft_labels = state.optimizable_params[0]
            predicted_labels = torch.argmax(soft_labels, dim=1)
            probabilities = torch.softmax(soft_labels, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            suffix = (
                f"pred={predicted_labels.cpu().tolist()}  "
                f"conf={[f'{c:.2f}' for c in confidences.cpu().tolist()]}"
            )
        # If using latent representation (e.g., GIAS), log latent code statistics
        if self.representation is not None and hasattr(self.representation, "latent_dim"):
            # Extract latent code from optimizable_tensor [E, N, G, latent_dim]
            # For GIAS stage 1: take first seed (E=0), first image (N=0), first restart (G=0)
            if state.optimizable_tensor.ndim == 4:
                latent_code = state.optimizable_tensor[0, 0, 0]

        if self.verbose:
            suffix += f"  lr={current_lr:.6f}"
        prefix = f"Iteration {state.iteration}/{self.max_iterations}"
        log_progress(logger, prefix, total_loss, losses, suffix, latent_code=latent_code)

    def _check_early_stop(
        self,
        state: InternalOptimizerState,
        loss_value: float,
        losses: dict,
    ) -> bool:
        """Check early stopping and update best state."""
        if loss_value < state.best_loss:
            state.best_loss = loss_value
            state.best_optimizable_tensor = state.optimizable_tensor.detach().clone()

            # Get current effective labels
            best_effective_labels = self.label_strategy.get_labels_for_forward(
                state.optimizable_params, state.labels
            )
            state.best_labels = best_effective_labels.detach().clone() if isinstance(
                best_effective_labels, torch.Tensor
            ) else best_effective_labels

            # Store loss breakdown
            state.aux_data["best_losses"] = losses
            state.aux_data["stagnant_iterations"] = 0
            return False

        state.aux_data["stagnant_iterations"] = state.aux_data.get("stagnant_iterations", 0) + 1
        return state.aux_data["stagnant_iterations"] >= self.patience


__all__ = ["ComposableOptimizer", "InternalOptimizerState"]
