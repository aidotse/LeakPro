#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""CMA-ES driver for GGL: population-based, gradient-free latent search.

Evaluates each generation's candidates by decoding them through the
representation (under ``torch.no_grad()``) and scoring the stage's loss terms
through the shared evaluator with a first-order dummy signal — no
backpropagation through the generator or into the score anywhere in this
driver, matching the population-based nature of CMA-ES.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.events import StepEvent
from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback, Driver
from leakpro.attacks.gia_attacks.modular.core.loss_eval import evaluate_stage_losses
from leakpro.attacks.gia_attacks.modular.core.result import DriverResult, clone_artifacts

if TYPE_CHECKING:
    import numpy as np

    from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
    from leakpro.attacks.gia_attacks.modular.core.state import Context


@dataclass(frozen=True)
class CMAESDriver(Driver):
    """Owns the CMA-ES ask/tell loop for a single-parameter-group representation.

    Requires ``rep.parameters()`` to return exactly one param group with a
    single tensor (true of ``GANRepresentation``) — CMA-ES optimizes one flat
    real vector, so a representation exposing more than one tensor is not
    supported by this driver.

    Reproducibility: pycma samples from its own NumPy RNG, seeded with ``seed``. With ``seed=None`` (the
    default) the seed is drawn from the torch global RNG, which ``run_attack`` seeds from
    ``RunMetadata.seed``, so the same run seed gives the same search.

    Raises:
        ValueError: If ``seed`` is given and not a positive integer (pycma treats 0 as "seed from the clock").

    """

    population_size: int = 32
    sigma0: float = 0.5
    max_generations: int = 200
    seed: int | None = None

    def __post_init__(self) -> None:
        """Rejects a seed pycma would replace with the clock."""
        if self.seed is not None and self.seed < 1:
            raise ValueError(f"seed must be a positive integer, got {self.seed}")

    def run(
        self,
        stage: OptimStage,
        ctx: Context,
        *,
        callbacks: Sequence[Callback] = (),
        stage_idx: int = 0,
    ) -> DriverResult:
        """Run the CMA-ES ask/tell loop; the result is the best candidate's output, artifacts and fitness."""
        try:
            import cma  # noqa: PLC0415 -- optional dependency, imported lazily
        except ImportError as exc:
            raise ImportError(
                "CMAESDriver requires the `cma` package. Install it with: pip install cma"
            ) from exc

        rep, target_source = stage.rep, stage.target_source
        param: Tensor = rep.parameters()[0]["params"][0]
        shape = param.shape
        x0 = param.detach().cpu().numpy().flatten().astype("float64")
        target = target_source.current()

        def fitness(vector: np.ndarray) -> float:
            candidate = torch.tensor(vector, dtype=param.dtype, device=param.device).view(shape)
            with torch.no_grad():
                param.data.copy_(candidate)
                recon = rep.decode()
            # The dummy signal needs autograd (it differentiates the client loss w.r.t. the weights);
            # create_graph=False keeps it first-order, since CMA-ES never backpropagates the score.
            return evaluate_stage_losses(stage.losses, ctx, recon, target, create_graph=False).total.item()

        seed = self.seed if self.seed is not None else int(torch.randint(1, 2**31 - 1, (1,)).item())
        solver = cma.CMAEvolutionStrategy(
            x0, self.sigma0,
            {"popsize": self.population_size, "maxfevals": self.population_size * self.max_generations, "verbose": -9,
             "seed": seed},
        )

        best_vector = x0
        best_fitness = fitness(x0)

        generation = 0
        while not solver.stop() and generation < self.max_generations:
            solutions = solver.ask()
            fitnesses = [fitness(v) for v in solutions]
            solver.tell(solutions, fitnesses)

            gen_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_vector = solutions[gen_best_idx]

            for cb in callbacks:
                cb.on_step_end(StepEvent(stage_idx, generation, best_fitness, {"generation_best": min(fitnesses)},
                                          rep.artifacts()))
            generation += 1

        best_tensor = torch.tensor(best_vector, dtype=param.dtype, device=param.device).view(shape)
        with torch.no_grad():
            param.data.copy_(best_tensor)

        return DriverResult(
            input=rep.to_observable().clone(),
            target=target.detach().clone() if target is not None else None,
            artifacts=clone_artifacts(rep.artifacts()),
            loss=float(best_fitness),
            iterations=generation,
            slice_losses=None,
        )
