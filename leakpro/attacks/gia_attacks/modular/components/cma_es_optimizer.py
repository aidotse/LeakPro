#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer for gradient inversion.

This module implements a population-based evolutionary strategy for latent code
optimization, following the GGL (Gradient Gradient Leakage) paper approach.

The optimizer inherits from GradientInversionBase to reuse the scaffold
(seed/epoch aggregation, BN setup, state building) while providing a
completely different core optimization loop based on CMA-ES.

Key differences from gradient-based optimizers (Adam, LBFGS):
- Population-based sampling: evaluates multiple candidates per generation
- Gradient-free: uses only loss values, no backpropagation
- Adaptive covariance matrix: learns the fitness landscape structure
- KL divergence budget: controls search radius via explicit parameter
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional

import torch
from torch import nn

try:
    import cma
except ImportError:
    cma = None  # type: ignore

from leakpro.attacks.gia_attacks.modular.components.gradient_inversion_base import (
    GradientInversionBase,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.optimizer_utils import (
    compute_loss_components,
)
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    AggregationStrategy,
    ComponentMetadata,
    LabelInferenceResult,
    OptimizationState,
)
from leakpro.attacks.gia_attacks.modular.core.state import RunContext

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
        LossComponent,
    )
    from leakpro.attacks.gia_attacks.modular.components.representation_strategies import RepresentationStrategy

logger = logging.getLogger(__name__)


class CMAESOptimizer(GradientInversionBase):
    """CMA-ES optimizer for gradient inversion via population-based search.

    Uses Covariance Matrix Adaptation Evolution Strategy (Hansen & Ostermeier)
    to optimize latent codes directly without gradients. Particularly effective
    when combined with large-scale GANs (BigGAN, StyleGAN2) as it exploits
    the smooth landscape of GAN latent spaces.

    Args:
        loss_components: List of loss components computing the objective.
        loss_fn: FL loss function (default: CrossEntropyLoss).
        seed_aggregation: Strategy for aggregating across seeds.
        epoch_aggregation: Strategy for aggregating across epochs.
        log_interval: Log progress every n generations.
        max_iterations: Total number of CMA-ES function evaluations (budget).
        representation: Optional representation strategy for latent-space optimization.
        cma_population_size: Population size per generation. If None,
            automatically set to 4 + floor(3*log(dimension)).
        cma_kld: KL divergence budget parameter. Controls step size and
            exploration radius. (0.1 for BigGAN, 0.02 for StyleGAN2).
        cma_cost_fn: Loss function type ("l2", "cosine", "sim_cmpr0").

    """

    def __init__(
        self,
        loss_components: List[LossComponent],
        loss_fn: nn.Module | None = None,
        seed_aggregation: AggregationStrategy | None = None,
        epoch_aggregation: AggregationStrategy | None = None,
        log_interval: int = 100,
        max_iterations: int = 500,
        representation: Optional["RepresentationStrategy"] = None,
        cma_population_size: int | None = None,
        cma_kld: float = 0.1,
        cma_cost_fn: str = "l2",
    ) -> None:
        """Initialize CMA-ES optimizer."""
        if cma is None:
            raise ImportError(
                "CMA-ES optimizer requires 'cma' package. "
                "Install with: pip install cma"
            )

        super().__init__(
            loss_components=loss_components,
            loss_fn=loss_fn,
            seed_aggregation=seed_aggregation,
            epoch_aggregation=epoch_aggregation,
            log_interval=log_interval,
        )
        self.max_iterations = max_iterations
        self.representation = representation
        self.cma_population_size = cma_population_size
        self.cma_kld = cma_kld
        self.cma_cost_fn = cma_cost_fn

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for CMA-ES optimizer."""
        return ComponentMetadata(
            name="cma_es_optimizer",
            required_capabilities={"has_gradients": False},
        )

    def _run_core_loop(
        self,
        x_init: torch.Tensor,
        labels: LabelInferenceResult,
        target_grads: List[torch.Tensor],
        ctx: RunContext,
        *,
        stage_idx: int = 0,
        callbacks: list | None = None,
    ) -> OptimizationState:
        """Run CMA-ES optimization loop.

        Args:
            x_init: Initial parameter tensor [E, N, G, latent_dim] (latent-space)
                or [E, N, G, C, H, W] (pixel-space).
            labels: Inferred label result — .labels is [N] hard labels.
            target_grads: Detached target gradients from client.
            ctx: Immutable run context.

        Returns:
            OptimizationState with reconstruction in [E, N, G, C, H, W].

        """
        device = x_init.device
        dtype = x_init.dtype
        param_shape = x_init.shape  # [E, N, G, ...]

        # labels.labels arrives as [E, N] (epoch dim added by orchestrator).
        # strip_epoch_dim → [N] for loss computation; keep [E, N] for conditional GAN.
        gan_labels = labels.labels.to(device)            # [E, N]
        flat_labels = labels.label_type.strip_epoch_dim(gan_labels)  # [N]

        # Flatten parameter tensor to 1-D vector for CMA-ES.
        x0 = x_init.detach().cpu().reshape(-1).numpy().astype("float64")
        dim = len(x0)
        pop_size = self.cma_population_size or (4 + int(3 * math.log(dim)))

        solver = cma.CMAEvolutionStrategy(
            x0=x0,
            sigma0=self.cma_kld,
            inopts={
                "popsize": pop_size,
                "maxfevals": self.max_iterations,
                "verbose": -9,  # suppress cma-lib output
            },
        )

        best_loss = float("inf")
        best_solution = x0.copy()
        generation = 0

        while not solver.stop():
            solutions = solver.ask()
            fitness_values = []

            for solution in solutions:
                candidate = torch.tensor(
                    solution, dtype=dtype, device=device
                ).reshape(param_shape)

                # Decode to pixel space if a representation is used.
                if self.representation is not None:
                    with torch.no_grad():
                        reconstruction = self.representation.forward(candidate, labels=gan_labels)
                else:
                    reconstruction = candidate

                # Compute gradient matching loss (requires model parameter gradients).
                total_loss, _ = compute_loss_components(
                    self.loss_components,
                    reconstruction,
                    flat_labels,
                    target_grads,
                    ctx,
                )
                fitness_values.append(float(total_loss.detach().cpu()))

            solver.tell(solutions, fitness_values)

            current_best = min(fitness_values)
            if current_best < best_loss:
                best_loss = current_best
                best_solution = solutions[fitness_values.index(current_best)].copy()

            generation += 1
            if generation % self.log_interval == 0:
                logger.info(
                    "CMA-ES gen %d  best_loss=%.6f  evals=%d",
                    generation, best_loss, solver.countevals,
                )

        # Decode best solution.
        best_tensor = torch.tensor(
            best_solution, dtype=dtype, device=device
        ).reshape(param_shape)

        with torch.no_grad():
            if self.representation is not None:
                final_reconstruction = self.representation.forward(best_tensor, labels=gan_labels)
            else:
                final_reconstruction = best_tensor

        # Base class expects labels in [E, N]; it will strip to [N].
        final_labels = gan_labels  # already [E, N]

        logger.info(
            "CMA-ES finished: %d generations, %d evals, best_loss=%.6f",
            generation, solver.countevals, best_loss,
        )

        return OptimizationState(
            reconstruction=final_reconstruction,   # [E, N, G, C, H, W]
            labels=final_labels,                   # [E, N]
            loss=best_loss,
            converged=True,
            metrics={"best_loss": best_loss, "generations": generation},
        )
