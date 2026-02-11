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
from typing import Any, Dict, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.constraints import (
    ConstraintStrategy,
    NoConstraint,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.label_strategies import (
    FixedLabels,
    LabelStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
    BNStatisticsRegularization,
    LossComponent,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.step_strategies import (
    StandardStepStrategy,
    StepStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    TrainingSimulator,
)
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    Component,
    ComponentMetadata,
    LabelInferenceResult,
    OptimizationState,
    OptimizationStrategy,
    SeedAggregationStrategy,
)
from leakpro.fl_utils.fl_client_simulator import ClientObservations

logger = logging.getLogger(__name__)


@dataclass
class InternalOptimizerState:
    """Internal state maintained during ComposableOptimizer optimization."""

    client_observations: ClientObservations
    reconstruction: torch.Tensor
    labels: torch.Tensor
    optimizable_params: List[torch.Tensor]
    optimizer: optim.Optimizer
    scheduler: Any | None
    iteration: int
    best_loss: float
    best_reconstruction: torch.Tensor | None
    best_labels: torch.Tensor | None
    aux_data: Dict[str, Any]  # For storing intermediate results


class ComposableOptimizer(OptimizationStrategy):
    """Composable optimization strategy built from reusable building blocks.

    Args:
        loss_components: List of loss components to combine
        constraint: Constraint strategy to apply after each step
        label_strategy: How to handle labels during optimization
        step_strategy: How to execute optimization steps
        learning_rate: Base learning rate
        max_iterations: Number of optimization iterations
        optimizer_type: Type of optimizer ("adam", "sgd", "lbfgs")
        scheduler_type: Type of scheduler ("cosine", "step", "exponential", None)
        patience: Early stopping patience (iterations without improvement)

    """

    def __init__(
        self,
        loss_components: List[LossComponent],
        constraint: ConstraintStrategy | None = None,
        label_strategy: LabelStrategy | None = None,
        step_strategy: StepStrategy | None = None,
        learning_rate: float = 0.1,
        label_learning_rate: float | None = None,
        max_iterations: int = 300,
        optimizer_type: str = "adam",
        scheduler_type: str | None = None,
        patience: int = 10000,
        log_interval: int = None,
        training_simulator: TrainingSimulator | None = None,
        loss_fn: nn.Module | None = None,
        seed_aggregation: SeedAggregationStrategy | None = None,
    ) -> None:
        self.loss_components = loss_components
        self.constraint = constraint or NoConstraint()
        self.label_strategy = label_strategy or FixedLabels()
        self.step_strategy = step_strategy or StandardStepStrategy()
        self.learning_rate = learning_rate
        self.label_learning_rate = label_learning_rate if label_learning_rate is not None else learning_rate
        self.max_iterations = max_iterations
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.patience = patience
        self.log_interval = log_interval
        self._training_simulator = training_simulator
        self.seed_aggregation = seed_aggregation

        super().__init__()

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

        return ComponentMetadata(
            name="composable",
            display_name="Composable Optimizer",
            description="Flexible optimization built from composable building blocks",
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
        if isinstance(component, BNStatisticsRegularization):
            strategy_reqs = component.get_strategy_requirements()
            requirements.update(strategy_reqs)

        return requirements

    def optimize(
        self,
        reconstruction: torch.Tensor,
        labels: LabelInferenceResult,
        target_model: nn.Module,
        client_observations: ClientObservations,
        proxy_dataloader: DataLoader | None = None,
    ) -> OptimizationState:
        """Run optimization using composed building blocks."""
        observed_gradients = client_observations.gradients
        data_mean = client_observations.data_mean
        data_std = client_observations.data_std

        # Convert observed_gradients dict to list matching model.parameters()
        # Detach target gradients - they should not require gradients themselves
        target_gradients = [
            observed_gradients[name].detach()
            for name, _ in target_model.named_parameters()
            if name in observed_gradients
        ]

        # Setup BN regularization components if present
        self._setup_bn_components(target_model, reconstruction, client_observations, proxy_dataloader)

        # Setup optimization state
        state = self._setup_optimization(
            reconstruction, labels, client_observations
        )

        log_interval = self.log_interval if self.log_interval is not None else max(1, self.max_iterations // 10)
        logger.debug(f"Starting optimization: max_iters={self.max_iterations}, "
                     f"log_interval={log_interval}, patience={self.patience}")

        # Main optimization loop
        for iteration in range(self.max_iterations):
            state.iteration = iteration

            # Define closures for step strategy
            def compute_loss_fn() -> tuple[torch.Tensor, dict[str, float]]:
                return self._compute_loss(state, target_model, target_gradients, self.loss_fn)

            def apply_constraints_fn(s: InternalOptimizerState) -> None:
                with torch.no_grad():
                    s.reconstruction.data = self.constraint.apply(
                        s.reconstruction.data, data_mean, data_std
                    )

            # Execute optimization step using strategy
            total_loss_value, losses = \
                self.step_strategy.execute_step(state, compute_loss_fn, apply_constraints_fn)

            # Check for best reconstruction and early stopping
            should_stop = self._check_early_stop(
                state, total_loss_value, losses
            )

            if should_stop:
                logger.info(f"  Early stopping at iteration {iteration}")
                break

            # Logging
            if iteration % log_interval == 0 or iteration == self.max_iterations - 1:
                self._log_progress(state, total_loss_value, losses)

        # Return final best reconstruction
        if state.best_reconstruction is not None:
            final_reconstruction = state.best_reconstruction

            # Apply seed aggregation if multi-seed and aggregation strategy provided
            if self.seed_aggregation is not None and final_reconstruction.ndim == 5:
                logger.info(f"  Applying seed aggregation: {self.seed_aggregation.get_metadata().name}")
                final_reconstruction = self.seed_aggregation.compute_consensus(final_reconstruction)

            final_labels = state.best_labels if state.best_labels is not None else labels
            return OptimizationState(
                reconstruction=final_reconstruction,
                labels=final_labels,
                loss=state.best_loss,
                iteration=state.iteration,
                converged=(state.aux_data.get("stagnant_iterations", 0) < self.patience),
                metrics={
                    "gradient_loss": state.aux_data.get("best_gradient_loss", 0.0),
                    "regularization_loss": state.aux_data.get("best_regularization_loss", 0.0),
                },
            )

        # Fallback to current state
        final_reconstruction = state.reconstruction.detach()

        # Apply seed aggregation if multi-seed and aggregation strategy provided
        if self.seed_aggregation is not None and final_reconstruction.ndim == 5:
            logger.info(f"  Applying seed aggregation: {self.seed_aggregation.get_metadata().name}")
            final_reconstruction = self.seed_aggregation.compute_consensus(final_reconstruction)

        current_labels = self.label_strategy.get_labels_for_forward(
            state.optimizable_params, state.labels
        )
        return OptimizationState(
            reconstruction=final_reconstruction,
            labels=current_labels if isinstance(current_labels, torch.Tensor) else labels,
            loss=total_loss_value,
            iteration=state.iteration,
            converged=False,
        )

    def _setup_bn_components(
        self,
        model: nn.Module,
        reconstruction: torch.Tensor,
        client_observations: ClientObservations,
        proxy_dataloader: DataLoader | None = None,
    ) -> None:
        """Setup BN regularization components if present.

        Looks for BNStatisticsRegularization in loss_components and
        calls setup() with the model and client observations.
        Uses the optimizer's canonical training_simulator.

        Args:
            model: Target model
            reconstruction: Current reconstruction
            client_observations: ClientObservations containing BN statistics
            proxy_dataloader: Optional server-side dataloader for ProxyBNStatisticsStrategy

        """
        for component in self.loss_components:
            if isinstance(component, BNStatisticsRegularization):
                strategy_name = component.strategy.get_metadata().name
                logger.info(f"  Setting up {component.name} with strategy: {strategy_name}")
                component.setup(
                    model=model,
                    reconstruction=reconstruction,
                    client_observations=client_observations,
                    training_simulator=self._training_simulator,
                    proxy_dataloader=proxy_dataloader,
                )

    def _setup_optimization(
        self,
        reconstruction: torch.Tensor,
        labels: LabelInferenceResult,
        client_observations: ClientObservations,
    ) -> InternalOptimizerState:
        """Setup optimization state and parameters."""

        effective_labels, label_params = self.label_strategy.setup(labels)

        # Log label strategy
        strategy_name = self.label_strategy.__class__.__name__
        if label_params:
            soft_labels = label_params[0]
            predicted_labels = torch.argmax(soft_labels, dim=1)
            probabilities = torch.softmax(soft_labels, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            logger.info(
                f"  Label Strategy: {strategy_name} (joint optimization)\n"
                f"    Initial predictions: {predicted_labels.cpu().tolist()} "
                f"with confidence: {[f'{c:.3f}' for c in confidences.cpu().tolist()]}"
            )
        else:
            logger.info(f"  Label Strategy: {strategy_name} (fixed labels: {effective_labels.cpu().tolist()})")

        # Parameters to optimize
        if not reconstruction.is_leaf:
            reconstruction = reconstruction.detach().clone()
        reconstruction.requires_grad = True

        # Create optimizer with separate parameter groups for image and labels
        if label_params:
            # Use separate learning rates for image and label parameters
            param_groups = [
                {"params": [reconstruction], "lr": self.learning_rate},
                {"params": label_params, "lr": self.label_learning_rate}
            ]
            logger.info(f"  Using separate learning rates: image_lr={self.learning_rate}, label_lr={self.label_learning_rate}")
        else:
            # Only image parameters, single learning rate
            param_groups = [{"params": [reconstruction], "lr": self.learning_rate}]

        # Create optimizer
        if self.optimizer_type == "adam":
            optimizer = optim.Adam(param_groups)
        elif self.optimizer_type == "lbfgs":
            optimizer = optim.LBFGS(
                param_groups,
                lr=self.learning_rate,
                max_iter=20,
                history_size=100,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        # Create scheduler
        scheduler = None
        if self.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_iterations
            )
        elif self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[self.max_iterations // 2.667,
                                                                        self.max_iterations // 1.6,
                                                                        self.max_iterations // 1.142], gamma=0.1)
        elif self.scheduler_type == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        else:
            scheduler = None  # No scheduler

        return InternalOptimizerState(
            client_observations=client_observations,
            reconstruction=reconstruction,
            labels=effective_labels,
            optimizable_params=label_params,
            optimizer=optimizer,
            scheduler=scheduler,
            iteration=0,
            best_loss=float("inf"),
            best_reconstruction=None,
            best_labels=None,
            aux_data={},
        )

    def _compute_loss(
        self,
        state: InternalOptimizerState,
        model: nn.Module,
        target_gradients: List[torch.Tensor],
        loss_fn: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss from all components.

        Args:
            state: Current optimization state
            model: Target model
            target_gradients: True gradients to match
            loss_fn: Loss function for gradient computation

        Returns:
            Total loss and dict of individual component losses

        """
        # Get effective labels for this iteration
        effective_labels = self.label_strategy.get_labels_for_forward(
            state.optimizable_params, state.labels
        )
        losses = {}

        for component in self.loss_components:
            component_loss = component.compute(
                reconstruction=state.reconstruction,
                model=model,
                labels=effective_labels,
                target_gradients=target_gradients,
                loss_fn=loss_fn,
            )
            losses[component.name] = component_loss

        total_loss = sum(losses.values())
        return total_loss, losses

    def _log_progress(
        self,
        state: InternalOptimizerState,
        total_loss: float,
        losses: dict[str, float],
    ) -> None:
        """Log optimization progress."""
        log_msg = f"  Iteration {state.iteration}/{self.max_iterations}: "
        log_msg += f"total_loss={total_loss:.4f} "
        for name, loss_value in losses.items():
            log_msg += f"{name}={loss_value:.4f} "

        # Show predicted labels if joint optimization
        if state.optimizable_params:
            soft_labels = state.optimizable_params[0]
            predicted_labels = torch.argmax(soft_labels, dim=1)
            probabilities = torch.softmax(soft_labels, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            log_msg += f", pred={predicted_labels.cpu().tolist()}"
            log_msg += f", conf={[f'{c:.2f}' for c in confidences.cpu().tolist()]}"

        logger.info(log_msg)

    def _check_early_stop(
        self,
        state: InternalOptimizerState,
        loss_value: float,
        losses: dict,
    ) -> bool:
        """Check early stopping and update best state."""
        if loss_value < state.best_loss:
            state.best_loss = loss_value
            state.best_reconstruction = state.reconstruction.detach().clone()

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
