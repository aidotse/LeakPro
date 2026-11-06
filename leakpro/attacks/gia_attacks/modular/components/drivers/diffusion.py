#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Diffusion-sampling driver for GGDM/GradInvDiff.

Replaces gradient descent with a guided reverse-diffusion loop over a
pre-trained UNet denoiser.

The sample x_t is held by a ``DiffusionRepresentation`` in the denoiser's [-1, 1] space; the
driver overwrites its tensor each reverse step via ``.data.copy_()``. The guidance objective is
evaluated in data space, at ``rep.to_data(x_t)``, like every other driver's objective.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.representations.diffusion import DiffusionRepresentation
from leakpro.attacks.gia_attacks.modular.core.events import StepEvent
from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback, Driver
from leakpro.attacks.gia_attacks.modular.core.loss_eval import evaluate_stage_losses
from leakpro.attacks.gia_attacks.modular.core.result import DriverResult, clone_artifacts

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.diffusion.mean_strategies import MeanAdjustmentStrategy
    from leakpro.attacks.gia_attacks.modular.components.diffusion.noise_strategies import NoiseInjectionStrategy
    from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
    from leakpro.attacks.gia_attacks.modular.core.state import Context
    from leakpro.fl_utils.diffusion_handler import DDPMSchedulerWrapper


Sampler = Literal["ddpm", "ddim"]


@dataclass(frozen=True)
class DiffusionSamplingDriver(Driver):
    """Owns the reverse-diffusion loop (t = T-1 ... 0) for a DiffusionRepresentation.

    Args:
        unet: Frozen denoising UNet (e.g. from ``load_pretrained_ddpm``);
            called as ``unet(xt, t_tensor).sample`` to predict noise.
        scheduler: A ``DDPMSchedulerWrapper`` exposing ``num_timesteps``,
            ``predict_x0_from_eps``, ``step_mean``, ``posterior_std``.
        mean_strategy: Shifts the raw step mean toward the matching objective
            each step (``SimilarityGuidance`` for GGDM,
            ``AdaptiveMeanOptimization`` for GradInvDiff).
        noise_strategy: Samples the stochastic noise term for DDPM steps
            (unused entirely for ``sampler="ddim"``, which is deterministic).
        sampler: ``"ddpm"`` (stochastic ancestral) or ``"ddim"`` (deterministic).
        num_steps: Number of reverse steps to actually take. ``None`` runs
            every integer timestep (``T`` steps); a smaller value subsamples
            evenly-spaced waypoints and takes exactly ``num_steps`` reverse steps (each one UNet
            evaluation). With ``sampler="ddpm"`` each step samples the posterior of the actual jump
            ``t → t_prev``, so the sample stays at the noise level of the next waypoint.

    """

    unet: Any
    scheduler: DDPMSchedulerWrapper
    mean_strategy: MeanAdjustmentStrategy
    noise_strategy: NoiseInjectionStrategy
    sampler: Sampler = "ddpm"
    num_steps: int | None = None

    def run(
        self,
        stage: OptimStage,
        ctx: Context,
        *,
        callbacks: Sequence[Callback] = (),
        stage_idx: int = 0,
    ) -> DriverResult:
        """Run the reverse-diffusion loop and return the final decoded sample.

        Step events and the result report a NaN loss: reverse-diffusion sampling has no per-step
        objective.
        """
        rep, target_source = stage.rep, stage.target_source
        if not isinstance(rep, DiffusionRepresentation):
            raise TypeError(f"DiffusionSamplingDriver requires a DiffusionRepresentation, got {type(rep).__name__}")
        param: Tensor = rep.parameters()[0]["params"][0]
        num_timesteps = self.scheduler.num_timesteps
        target = target_source.current()

        def objective(x_t: Tensor) -> Tensor:
            # The stage terms score data space; the guidance gradient flows back through to_data.
            return evaluate_stage_losses(stage.losses, ctx, rep.to_data(x_t), target, full_recon=None).total

        with torch.no_grad():
            param.data.copy_(torch.randn_like(param))

        waypoints = self._build_waypoints(num_timesteps)
        step_pairs = list(zip(waypoints, waypoints[1:] + [-1]))

        for step_idx, (t, t_prev) in enumerate(step_pairs):
            xt = param.detach()
            t_tensor = torch.full((xt.shape[0],), t, device=ctx.device, dtype=torch.long)

            with torch.no_grad():
                eps_pred = self.unet(xt, t_tensor).sample

            x0_pred = self.scheduler.predict_x0_from_eps(xt, t, eps_pred).clamp(-1.0, 1.0)

            mu_theta = (
                x0_pred if t_prev < 0
                else self.scheduler.step_mean(x0_pred, xt, eps_pred, t, t_prev, sampler=self.sampler)
            )

            sigma_t = self.scheduler.posterior_std(t, t_prev).item() if self.sampler == "ddpm" else 0.0

            adjusted_mean, delta_mu = self.mean_strategy.adjust(
                mu_theta, xt, sigma_t, t, num_timesteps, objective
            )

            if self.sampler == "ddpm" and t_prev >= 0:
                noise = self.noise_strategy.sample(xt.shape, xt.device, xt.dtype, delta_mu)
                x_next = adjusted_mean + sigma_t * noise
            else:
                x_next = adjusted_mean

            with torch.no_grad():
                param.data.copy_(x_next)

            for cb in callbacks:
                cb.on_step_end(StepEvent(stage_idx, step_idx, float("nan"), {}, rep.artifacts()))

        return DriverResult(
            input=rep.to_observable().clone(),
            target=target.detach().clone() if target is not None else None,
            artifacts=clone_artifacts(rep.artifacts()),
            loss=float("nan"),
            iterations=len(step_pairs),
            slice_losses=None,
        )

    def _build_waypoints(self, num_timesteps: int) -> list[int]:
        """Descending list of timesteps to visit, ending at 0.

        ``None`` (or a requested count that covers every timestep) visits every
        integer ``t`` from ``T-1`` down to ``0`` — ``T`` waypoints, i.e. ``T``
        reverse steps once paired with the trailing ``-1`` sentinel below.

        A smaller ``num_steps`` picks exactly ``num_steps`` evenly-spaced
        waypoints spanning ``[0, T-1]`` (always including both endpoints), so
        that pairing them with the trailing ``-1`` sentinel yields exactly
        ``num_steps`` reverse steps — matching the ``None`` case's convention
        that the waypoint count equals the executed step count.
        """
        if self.num_steps is None or self.num_steps >= num_timesteps:
            return list(range(num_timesteps - 1, -1, -1))

        if self.num_steps <= 1:
            return [num_timesteps - 1]

        return sorted(
            {round((num_timesteps - 1) * i / (self.num_steps - 1)) for i in range(self.num_steps)},
            reverse=True,
        )
