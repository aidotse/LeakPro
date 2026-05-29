#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Building blocks for diffusion-based gradient inversion attacks.

This module provides composable components for constructing diffusion-based
GIA methods.  The design follows the same philosophy as the rest of the
modular framework: small, single-responsibility building blocks that can be
mixed and matched.

Building blocks
---------------

**Mean adjustment strategies** — how gradient information modifies mu_theta:

* :class:`SimilarityGuidance` — add ``gamma * sigma^2 * grad_{x_t} sim`` to mu_theta
  (GGDM, Gu et al. 2024).
* :class:`AdaptiveMeanOptimization` — ``K`` inner Adam steps on a
  learnable mu*, then blend with mu_theta via a schedule (GradInvDiff, Wang et al. 2024).

**Noise injection strategies** — what noise to add at each reverse step:

* :class:`StandardNoiseInjection` — eps ~ N(0, I)  (classic DDPM / GGDM).
* :class:`GradientAlignedNoiseInjection` — project eps onto the
  gradient-residual direction ``delta_mu = mu* - mu_theta``  (GANI from GradInvDiff).

**Blending schedules** — time-dependent weight for mean blending:

* :class:`ConstantSchedule` — fixed gamma (used by SimilarityGuidance).
* :class:`LinearDecaySchedule` — gamma_t = t / (T-1), decays from 1->0.
* :class:`CosineDecaySchedule` — cosine annealing from 1->0.

**Optimizer** — the reverse-process loop that composes the above:

* :class:`DiffusionSamplingOptimizer` — runs the full T-step reverse
  sampling, delegating to the pluggable strategies at each step.

Composing attacks
-----------------

.. code-block:: python

    # GGDM (Gu et al.)
    optimizer = DiffusionSamplingOptimizer(
        unet=unet, scheduler=sched,
        mean_strategy=SimilarityGuidance(gamma=100.0, grad_clip=1.0),
        noise_strategy=StandardNoiseInjection(),
    )

    # GradInvDiff (Wang et al.)
    optimizer = DiffusionSamplingOptimizer(
        unet=unet, scheduler=sched,
        mean_strategy=AdaptiveMeanOptimization(
            inner_steps=5, inner_lr=0.01,
            schedule=LinearDecaySchedule(),
        ),
        noise_strategy=GradientAlignedNoiseInjection(),
    )

    # Mix-and-match: AMO guidance with standard noise
    optimizer = DiffusionSamplingOptimizer(
        unet=unet, scheduler=sched,
        mean_strategy=AdaptiveMeanOptimization(
            inner_steps=5, inner_lr=0.01,
            schedule=CosineDecaySchedule(),
        ),
        noise_strategy=StandardNoiseInjection(),
    )

References
----------
* Gu et al., "Federated Learning Vulnerabilities: Privacy Attacks with
  Denoising Diffusion Probabilistic Models", WWW 2024.
* Wang et al., "GradInvDiff: Stealing Medical Privacy in Federated
  Learning via Diffusion-Based Gradient Inversion", 2024.

"""

from __future__ import annotations

import logging
from typing import Any, List

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.gradient_inversion_base import GradientInversionBase

# Building blocks now live in optimization_building_blocks/ — imported here
# and re-exported so existing ``from diffusion_optimizer import X`` calls keep working.
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.blending_schedules import (
    BlendingSchedule,
    ConstantSchedule,
    CosineDecaySchedule,
    LinearDecaySchedule,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.diffusion_utils import (
    denorm_ddpm_to_model_space,
    norm_model_space_to_ddpm,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
    GradientMatchingLoss,
    LossComponent,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.mean_strategies import (
    AdaptiveMeanOptimization,
    MeanAdjustmentStrategy,
    SimilarityGuidance,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.noise_strategies import (
    GradientAlignedNoiseInjection,
    NoiseInjectionStrategy,
    StandardNoiseInjection,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.optimizer_utils import (
    log_progress,
)
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    AggregationStrategy,
    ComponentMetadata,
    LabelInferenceResult,
    OptimizationState,
)
from leakpro.attacks.gia_attacks.modular.core.state import RunContext

logger = logging.getLogger(__name__)


# ========================================================================= #
# Main optimizer — composes the building blocks                             #
# ========================================================================= #

class DiffusionSamplingOptimizer(GradientInversionBase):
    """Gradient inversion via guided DDPM reverse sampling.

    This optimizer runs the full ``T``-step reverse process, delegating
    the gradient-dependent parts to pluggable building blocks:

    At each timestep ``t = T-1 ... 0``:

    1. Predict eps_theta(x_t, t) via the frozen UNet -> derive mu_theta, sigma_t.
    2. Adjust the mean:  ``mu_adj, delta_mu = mean_strategy.adjust(...)``.
    3. Sample noise:     ``eps = noise_strategy.sample(..., delta_mu)``.
    4. Step:             ``x_{t-1} = mu_adj + sigma_t * eps``  (or mu_adj if t = 0).

    Args:
        unet: Pre-trained UNet denoiser (frozen, eval mode).
        scheduler: :class:`~leakpro.fl_utils.diffusion_handler.DDPMSchedulerWrapper`.
        mean_strategy: How to adjust mu_theta using gradient information.
        noise_strategy: How to sample noise at each step.
        loss_fn: FL loss function (default: ``CrossEntropyLoss``).
        loss_components: Loss components used to compute the guidance scalar at each
            reverse step — identical interface to
            :class:`~leakpro.attacks.gia_attacks.modular.components.composable_optimizer
            .ComposableOptimizer`.  Default: ``[GradientMatchingLoss("cosine")]``.
            The training simulator embedded in each :class:`GradientMatchingLoss` is
            injected by :meth:`~leakpro.attacks.gia_attacks.modular.configs.AttackConfig
            .build` so that client training settings (epochs, model mode, batch size)
            are respected automatically.
        log_interval: Print progress every *n* timesteps.

    """

    def __init__(
        self,
        unet: nn.Module,
        scheduler: Any,  # DDPMSchedulerWrapper
        mean_strategy: MeanAdjustmentStrategy,
        noise_strategy: NoiseInjectionStrategy | None = None,
        loss_fn: nn.Module | None = None,
        loss_components: List[LossComponent] | None = None,
        log_interval: int = 100,
        start_t_frac: float | None = None,
        stop_t_frac: float | None = None,
        sampler: str = "ddpm",
        num_steps: int | None = None,
        seed_aggregation: AggregationStrategy | None = None,
        epoch_aggregation: AggregationStrategy | None = None,
    ) -> None:
        if sampler not in ("ddpm", "ddim"):
            raise ValueError(f"sampler must be 'ddpm' or 'ddim', got '{sampler}'")
        if num_steps is not None and sampler != "ddim":
            raise ValueError(
                "num_steps subsampling is only valid with sampler='ddim'.  "
                "DDPM posterior coefficients assume t_prev = t-1 and are not "
                "correct for larger jumps."
            )
        # Delegate shared fields (loss_components, loss_fn, aggregation, log_interval)
        _loss_components = loss_components or [GradientMatchingLoss(loss_type="cosine")]
        super().__init__(_loss_components, loss_fn, seed_aggregation, epoch_aggregation, log_interval)
        self.unet = unet
        self.scheduler = scheduler
        self.mean_strategy = mean_strategy
        self.noise_strategy = noise_strategy or StandardNoiseInjection()
        self.start_t_frac = start_t_frac
        self.stop_t_frac = stop_t_frac
        # "ddpm": ancestral sampling (stochastic, σ_t·ε added at each step).
        # "ddim": deterministic ODE step (η=0, no noise).  Use with DDIM
        #         inversion cycling so inversion and decode are on the same ODE.
        self.sampler = sampler
        # Number of UNet evaluations per reverse pass.  None = every integer
        # timestep in [stop_t, start_t] (full resolution).  Values < the range
        # length subsample the ODE trajectory evenly (faster cycles, lower quality).
        # Only valid with sampler='ddim' (DDPM coefficients require unit steps).
        self.num_steps = num_steps

    # ------------------------------------------------------------------ #
    # OptimizationStrategy interface                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        return ComponentMetadata(
            name="diffusion_sampling_optimizer",
            required_capabilities={"has_gradients": True},
        )

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
        """Run the full DDPM reverse-process attack (seed-aware).

        ``x_init`` arrives as ``[E, N, G, C, H, W]`` from the base class scaffold.
        All ``N*G`` variants are processed in parallel through the UNet.
        The base class then applies seed/epoch aggregation and strips the dims.
        """
        target_model = ctx.target_model
        client_observations = ctx.client_observations
        device = next(target_model.parameters()).device
        data_mean = client_observations.data_mean
        data_std = client_observations.data_std

        # --- Image shape ------------------------------------------------
        N, C, H, W = client_observations.input_shape

        # --- Unpack seeds from 6D input ---------------------------------
        if x_init.ndim == 6:
            _E, _N, G = x_init.shape[:3]
        else:
            G = 1
        # Flatten to [N*G, C, H, W] for batched UNet calls
        if x_init.ndim == 6:
            xt = x_init[0].reshape(N * G, C, H, W).to(device).detach().clone()
        else:
            xt = x_init.to(device).detach().clone()

        # --- Labels -------------------------------------------------------
        lbl_base = (
            labels.labels.to(device)
            if labels is not None
            else torch.zeros(N, dtype=torch.long, device=device)
        )
        if lbl_base.ndim == 2:
            lbl_base = lbl_base[0]  # strip epoch dim [E, N] -> [N]
        # Repeat each label G times so lbl_flat aligns with [N*G] batch
        lbl_flat = lbl_base.repeat_interleave(G) if G > 1 else lbl_base

        # --- Reverse sampling loop ----------------------------------------
        T = self.scheduler.num_timesteps
        start_t = max(0, round(self.start_t_frac * T) - 1) if self.start_t_frac is not None else T - 1
        stop_t  = round(self.stop_t_frac  * T) if self.stop_t_frac  is not None else 0

        best_loss = float("inf")
        last_x0_pred = xt  # fallback if no step is ever executed

        loop_lower = stop_t + 1 if stop_t > 0 else 0
        total_range = start_t - loop_lower + 1
        if self.num_steps is not None and self.num_steps < total_range:
            n = self.num_steps
            ts_src = sorted(
                {round(loop_lower + (start_t - loop_lower) * i / (n - 1)) for i in range(n)},
                reverse=True,
            )
        else:
            ts_src = list(range(loop_lower, start_t + 1))[::-1]
        step_pairs = list(zip(ts_src, list(ts_src[1:]) + [loop_lower - 1]))

        if self.num_steps is not None and self.num_steps < total_range:
            pass

        for step_idx, (t, t_prev) in enumerate(step_pairs):
            xt, info = self._reverse_step(
                xt, t, t_prev, T, target_model, lbl_flat, target_grads,
                data_mean, data_std, ctx,
                n_images=N, n_seeds=G,
            )
            last_x0_pred = info.get("x0_pred", xt)

            step_loss = info.get("total_loss", float("inf"))
            best_loss = min(best_loss, step_loss)

            if step_idx % self.log_interval == 0 or step_idx == len(step_pairs) - 1:
                component_losses = {
                    k: v for k, v in info.items() if k not in ("total_loss", "x0_pred")
                }
                log_progress(logger, f"t={t:4d}/{T}", step_loss, component_losses)

        # --- Build output tensors -----------------------------------------
        x_stop_ddpm = xt.detach()  # [N*G, C, H, W]
        x0_est = last_x0_pred.detach() if isinstance(last_x0_pred, torch.Tensor) else x_stop_ddpm
        x_recon_flat = denorm_ddpm_to_model_space(x0_est, data_mean, data_std)

        # Reshape back to [1, N, G, C, H, W] for the base class scaffold
        x_recon = x_recon_flat.reshape(1, N, G, C, H, W)

        if G > 1:
            optimizable = x_stop_ddpm.reshape(N, G, C, H, W).mean(dim=1)  # [N, C, H, W]
        else:
            optimizable = x_stop_ddpm  # [N, C, H, W]

        lbl_out = lbl_base.unsqueeze(0)  # [1, N]

        return OptimizationState(
            reconstruction=x_recon,
            optimizable_tensor=optimizable,
            labels=lbl_out,
            loss=best_loss,
            iteration=start_t - loop_lower + 1,
            converged=True,
            metrics={"best_loss": best_loss, "stop_t": stop_t, "start_t": start_t},
        )

    # ------------------------------------------------------------------ #
    # Single reverse step — composes the building blocks                  #
    # ------------------------------------------------------------------ #

    def _reverse_step(
        self,
        xt: torch.Tensor,
        t: int,
        t_prev: int,
        T: int,
        model: nn.Module,
        labels: torch.Tensor,
        target_grads: list[torch.Tensor],
        data_mean: torch.Tensor | None,
        data_std: torch.Tensor | None,
        ctx: RunContext,
        n_images: int = 1,
        n_seeds: int = 1,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Execute one reverse-process step.

        1. UNet -> eps_theta -> mu_theta, sigma_t
        2. mean_strategy.adjust(...)  -> adjusted_mean, delta_mu
        3. noise_strategy.sample(...) -> noise
        4. x_{t_prev} = adjusted_mean + sigma_t * noise  (DDPM)
           x_{t_prev} = adjusted_mean                    (DDIM, no noise)
        """
        sched = self.scheduler

        # 1) UNet noise prediction  eps_theta(x_t, t)
        with torch.no_grad():
            t_tensor = torch.full(
                (xt.shape[0],), t, device=xt.device, dtype=torch.long,
            )
            eps_pred = self.unet(xt, t_tensor).sample

        # Derive posterior statistics
        x0_pred = sched.predict_x0_from_eps(xt, t, eps_pred)
        x0_pred = x0_pred.clamp(-1, 1)
        # sigma_t is used for guidance scaling in both DDPM and DDIM modes.
        sigma_t = sched.posterior_std(t)
        # Base mean depends on sampler:
        #   ddpm — stochastic DDPM posterior mean (Tweedie formula, valid for t_prev = t-1)
        #   ddim — deterministic ODE mean (η=0), supports any t_prev (subsampled steps)
        if self.sampler == "ddim":
            mu_theta = sched.ddim_step_mean(x0_pred, eps_pred, t, t_prev=t_prev)
        else:
            mu_theta = sched.posterior_mean(x0_pred, xt, t)

        # 2) Mean adjustment (gradient-guided)
        # sampler, t_prev, and eps_pred are forwarded so that strategies can
        # call scheduler.step_mean(x0_star, xt, eps_pred, t, t_prev, sampler)
        # without embedding any DDPM/DDIM step logic themselves.
        adjusted_mean, delta_mu, info = self.mean_strategy.adjust(
            mu_theta=mu_theta,
            sigma_t=sigma_t,
            xt=xt,
            x0_pred=x0_pred,
            scheduler=sched,
            t=t,
            T=T,
            ctx=ctx,
            labels=labels,
            target_grads=target_grads,
            loss_components=self.loss_components,
            n_images=n_images,
            n_seeds=n_seeds,
            sampler=self.sampler,
            t_prev=t_prev,
            eps_pred=eps_pred,
            unet=self.unet,
        )

        # 3) Noise injection
        # DDIM (η=0) is deterministic — no noise added at any step.
        # DDPM adds ancestral noise at all steps except t=0.
        if self.sampler == "ddpm" and t > 0:
            noise = self.noise_strategy.sample(
                shape=xt.shape,
                delta_mu=delta_mu,
                device=xt.device,
            )
            xt_prev = adjusted_mean + sigma_t * noise
        else:
            xt_prev = adjusted_mean

        info["x0_pred"] = x0_pred.detach()
        return xt_prev, info

    def __repr__(self) -> str:
        return (
            f"DiffusionSamplingOptimizer(\n"
            f"  mean_strategy={self.mean_strategy},\n"
            f"  noise_strategy={self.noise_strategy},\n"
            f"  sampler={self.sampler!r},\n"
            f"  num_steps={self.num_steps},\n"
            f"  start_t_frac={self.start_t_frac},\n"
            f"  stop_t_frac={self.stop_t_frac},\n"
            f"  loss_components={self.loss_components},\n"
            f")"
        )


__all__ = [
    # Schedules
    "BlendingSchedule",
    "ConstantSchedule",
    "LinearDecaySchedule",
    "CosineDecaySchedule",
    # Mean adjustment
    "MeanAdjustmentStrategy",
    "SimilarityGuidance",
    "AdaptiveMeanOptimization",
    # Noise injection
    "NoiseInjectionStrategy",
    "StandardNoiseInjection",
    "GradientAlignedNoiseInjection",
    # Optimizers
    "DiffusionSamplingOptimizer",
    # Helpers (public for custom strategies)
    "denorm_ddpm_to_model_space",
    "norm_model_space_to_ddpm",
]
