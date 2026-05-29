#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Utilities for loading and managing diffusion models for gradient inversion attacks.

This module provides functions to load pre-trained DDPM models via the
``diffusers`` library and exposes lightweight helpers that the
:class:`DiffusionSamplingOptimizer` uses during the reverse-process loop.

The design mirrors :mod:`leakpro.fl_utils.gan_handler` — a thin wrapper
that isolates the third-party dependency and keeps attack code clean.

Requires:
    pip install diffusers accelerate
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public: load a pre-trained DDPM
# ---------------------------------------------------------------------------

def load_pretrained_ddpm(
    model_id: str = "google/ddpm-cifar10-32",
    device: torch.device | str = "cpu",
    **kwargs: Any,
) -> tuple[nn.Module, "DDPMSchedulerWrapper"]:
    """Load a pre-trained unconditional DDPM from HuggingFace.

    Args:
        model_id: HuggingFace model ID.  Popular choices:
            - ``"google/ddpm-cifar10-32"``   (32×32, 3-channel, CIFAR-10)
            - ``"google/ddpm-celebahq-256"`` (256×256, 3-channel, CelebA-HQ)
            - ``"google/ddpm-cat-256"``      (256×256, 3-channel, LSUN Cat)
        device: Device to load the model on.
        **kwargs: Forwarded to ``DDPMPipeline.from_pretrained``.

    Returns:
        Tuple of ``(unet, scheduler_wrapper)`` where *unet* is the
        denoising UNet in eval mode and *scheduler_wrapper* exposes the
        noise schedule parameters needed by the attack.

    Raises:
        ImportError: If ``diffusers`` is not installed.
    """
    try:
        from diffusers import DDPMPipeline, DDPMScheduler  # noqa: F811
    except ImportError as exc:
        raise ImportError(
            "The `diffusers` library is required for diffusion-based GIA "
            "attacks.  Install it with:  pip install diffusers accelerate"
        ) from exc

    device = torch.device(device) if isinstance(device, str) else device

    logger.info("Loading pre-trained DDPM from HuggingFace: %s", model_id)
    pipeline = DDPMPipeline.from_pretrained(model_id, **kwargs)

    unet: nn.Module = pipeline.unet  # type: ignore[assignment]
    scheduler: DDPMScheduler = pipeline.scheduler  # type: ignore[assignment]

    # Freeze UNet — we never train it, only query it for μ_θ / ε_θ.
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False
    unet = unet.to(device)

    wrapper = DDPMSchedulerWrapper(scheduler, device=device)

    logger.info("✓ DDPM loaded  (UNet params: %s)", f"{sum(p.numel() for p in unet.parameters()):,}")
    return unet, wrapper


# ---------------------------------------------------------------------------
# Scheduler wrapper — exposes pre-computed noise-schedule tensors
# ---------------------------------------------------------------------------

class DDPMSchedulerWrapper:
    """Thin wrapper around a ``diffusers.DDPMScheduler``.

    Pre-computes and caches the noise-schedule tensors that the diffusion
    sampling loop needs, so attack code never imports ``diffusers`` directly.

    Attributes:
        num_timesteps: Total diffusion steps ``T``.
        betas:         β_t schedule  ``[T]``.
        alphas:        α_t = 1 − β_t  ``[T]``.
        alphas_cumprod: ᾱ_t = ∏_{s=1}^{t} α_s  ``[T]``.
        sqrt_alphas_cumprod:          √ᾱ_t  ``[T]``.
        sqrt_one_minus_alphas_cumprod: √(1 − ᾱ_t)  ``[T]``.
        posterior_variance:  β̃_t (for the reverse-process variance) ``[T]``.
    """

    def __init__(self, scheduler: Any, device: torch.device) -> None:
        self.num_timesteps: int = scheduler.config.num_train_timesteps
        self.device = device

        # Core schedule vectors (keep as float64 for numerical precision,
        # cast to float32 in the attack loop when needed).
        betas = scheduler.betas.to(device=device, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

        # Posterior variance  β̃_t = β_t · (1 − ᾱ_{t−1}) / (1 − ᾱ_t)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device, dtype=torch.float64), alphas_cumprod[:-1]])
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # Posterior mean coefficients (useful for computing μ̃_t from x_0 prediction)
        self.posterior_mean_coeff1 = betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        self.posterior_mean_coeff2 = (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)

    # ------------------------------------------------------------------ #
    # Convenience helpers used by the attack loop                         #
    # ------------------------------------------------------------------ #

    def q_sample(self, x0: torch.Tensor, t: int, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward-diffuse *x0* to timestep *t* (add noise).

        x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε

        Args:
            x0: Clean image(s)  ``[B, C, H, W]``.
            t: Timestep (0-indexed).
            noise: Optional pre-sampled noise; defaults to ``N(0, I)``.

        Returns:
            x_t with the same shape as *x0*.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t].float()
        b = self.sqrt_one_minus_alphas_cumprod[t].float()
        return a * x0 + b * noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: int, eps_pred: torch.Tensor) -> torch.Tensor:
        """Recover x̂_0 from the UNet noise prediction ε_θ(x_t, t).

        x̂_0 = (x_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t
        """
        a = self.sqrt_alphas_cumprod[t].float()
        b = self.sqrt_one_minus_alphas_cumprod[t].float()
        return (xt - b * eps_pred) / a

    def predict_eps_from_x0(self, xt: torch.Tensor, t: int, x0_pred: torch.Tensor) -> torch.Tensor:
        """Recover ε consistent with the pair (x_t, x̂_0).  Inverse of :meth:`predict_x0_from_eps`.

        Rearranging the forward diffusion equation
        ``x_t = √ᾱ_t · x̂_0 + √(1−ᾱ_t) · ε``  gives::

            ε* = (x_t − √ᾱ_t · x̂_0) / √(1−ᾱ_t)

        Use this after optimising x̂_0 so that the DDIM step uses the noise
        direction that is geometrically consistent with the updated x̂_0
        rather than the original UNet output ε_θ.
        """
        a = self.sqrt_alphas_cumprod[t].float()
        b = self.sqrt_one_minus_alphas_cumprod[t].float().clamp(min=1e-4)
        return (xt - a * x0_pred) / b

    def posterior_mean(self, x0_pred: torch.Tensor, xt: torch.Tensor, t: int) -> torch.Tensor:
        """Compute the posterior mean μ̃_t(x_t, x̂_0).

        μ̃_t = coeff1 · x̂_0  +  coeff2 · x_t
        """
        c1 = self.posterior_mean_coeff1[t].float()
        c2 = self.posterior_mean_coeff2[t].float()
        return c1 * x0_pred + c2 * xt

    def posterior_std(self, t: int) -> torch.Tensor:
        """Return σ_t = √β̃_t (posterior standard deviation)."""
        return self.posterior_variance[t].float().sqrt()

    def ddim_step_mean(
        self,
        x0_pred: torch.Tensor,
        eps_pred: torch.Tensor,
        t: int,
        t_prev: int | None = None,
    ) -> torch.Tensor:
        """DDIM deterministic step mean (eta=0): x0 -> x_{t_prev}.

        Supports both unit steps (t_prev = t - 1, default) and longer
        jumps for subsampled DDIM decode.  Pass t_prev < 0 as the signal
        for the final decode step — returns x0_pred directly.
        """
        if t_prev is None:
            t_prev = t - 1
        if t_prev < 0:
            return x0_pred
        a_prev = self.sqrt_alphas_cumprod[t_prev].float()
        b_prev = self.sqrt_one_minus_alphas_cumprod[t_prev].float()
        return a_prev * x0_pred + b_prev * eps_pred

    def step_mean(
        self,
        x0_pred: torch.Tensor,
        xt: torch.Tensor,
        eps_pred: torch.Tensor,
        t: int,
        t_prev: int,
        sampler: str = "ddpm",
    ) -> torch.Tensor:
        """Unified x̂_0 → step-mean for both DDPM and DDIM samplers.

        Single entry-point so all stepping logic lives here rather than
        scattered across mean-strategy code.  Both formulas are linear
        in *x0_pred*; swapping ``x0_pred`` for an optimised ``x̂_0*``
        produces the correct adjusted mean for either sampler.

        DDPM:  μ̃_t = c1[t] · x̂_0  +  c2[t] · x_t
        DDIM:  x_{t_prev} = √ᾱ_{t_prev} · x̂_0  +  √(1−ᾱ_{t_prev}) · ε_θ

        Args:
            x0_pred:  Clean-image estimate x̂_0  ``[B, C, H, W]``.
            xt:       Current noisy image x_t    ``[B, C, H, W]``
                      (used only for DDPM; ignored for DDIM).
            eps_pred: UNet noise prediction ε_θ  ``[B, C, H, W]``
                      (used only for DDIM; ignored for DDPM).
            t:        Current timestep.
            t_prev:   Target timestep (must be < *t*).  Pass ``-1`` for the
                      final decode step — DDIM returns *x0_pred* directly,
                      DDPM returns ``posterior_mean`` at ``t``.
            sampler:  ``"ddpm"`` (stochastic ancestral) or
                      ``"ddim"`` (deterministic η=0).

        Returns:
            Step-mean tensor, same shape as *x0_pred*.
        """
        if sampler == "ddim":
            return self.ddim_step_mean(x0_pred, eps_pred, t, t_prev=t_prev)
        return self.posterior_mean(x0_pred, xt, t)

    def step_mean_from_eps(
        self,
        eps_pred: torch.Tensor,
        xt: torch.Tensor,
        t: int,
        t_prev: int,
        sampler: str = "ddpm",
    ) -> torch.Tensor:
        """Compute the step mean directly from ε — numerically stable at all t.

        Equivalent to :meth:`step_mean` but avoids the intermediate
        ``x̂_0 = (x_t − b·ε) / a`` computation that blows up when
        ``a = √ᾱ_t → 0`` at high t.  Instead substitutes analytically::

            DDPM:  μ̃_t  = (c1/a + c2) · x_t  −  (c1·b/a) · ε
            DDIM:  x_{t_prev} = (a_prev/a) · x_t
                               + (b_prev − a_prev·b/a) · ε

        Both coefficients stay O(1) at all t because the ``1/a`` singularity
        is cancelled algebraically before evaluation.

        Use this (instead of ``step_mean``) whenever the optimisation variable
        is ε and you want to stay numerically safe across all timesteps.

        Args:
            eps_pred: Noise estimate ε  ``[B, C, H, W]``.
            xt:       Current noisy image x_t  ``[B, C, H, W]``.
            t:        Current timestep.
            t_prev:   Target timestep.  Pass ``-1`` for the final decode step
                      (t=0) — both branches return ``(xt − b·ε) / a.clamp``.
            sampler:  ``"ddpm"`` or ``"ddim"``.

        Returns:
            Step-mean tensor, same shape as *eps_pred*.
        """
        a = self.sqrt_alphas_cumprod[t].float()
        b = self.sqrt_one_minus_alphas_cumprod[t].float()
        a_safe = a.clamp(min=1e-4)  # only used in coefficient ratios — stays O(1)

        if t_prev < 0:
            # Final decode step: best approximation of x̂_0
            return (xt - b * eps_pred) / a_safe

        if sampler == "ddim":
            a_prev = self.sqrt_alphas_cumprod[t_prev].float()
            b_prev = self.sqrt_one_minus_alphas_cumprod[t_prev].float()
            ratio = a_prev / a_safe                        # √ᾱ_{t-1} / √ᾱ_t  ≈ 1 near t
            eps_coeff = b_prev - ratio * b                 # O(Δt) near consecutive steps
            return ratio * xt + eps_coeff * eps_pred

        # DDPM posterior mean expressed in ε
        c1 = self.posterior_mean_coeff1[t].float()
        c2 = self.posterior_mean_coeff2[t].float()
        xt_coeff = c1 / a_safe + c2    # (c1/a + c2) stays finite; c1 ∝ √ᾱ_{t-1} → 0 at high t
        eps_coeff = -(c1 * b) / a_safe # -(c1·b/a) stays finite by same cancellation
        return xt_coeff * xt + eps_coeff * eps_pred

    # ------------------------------------------------------------------ #
    # DDIM inversion — deterministic encoding x_0 → x_{t_inv}            #
    # ------------------------------------------------------------------ #

    def ddim_invert(
        self,
        x_in: torch.Tensor,
        t_end: int,
        unet: nn.Module,
        t_start: int = 0,
        num_steps: int | None = None,
        log_every: int = 50,
    ) -> torch.Tensor:
        """Forward DDIM ODE: deterministically encode x_{t_start} → x_{t_end}.

        Applies ``num_steps`` (default: ``t_end - t_start``, i.e. every integer
        step) deterministic forward ODE steps across evenly-spaced waypoints::

            x̂_0      = (x_t − √(1−ᾱ_t) ε_θ(x_t, t)) / √ᾱ_t
            x_{t_next} = √ᾱ_{t_next} · x̂_0 + √(1−ᾱ_{t_next}) · ε_θ(x_t, t)

        With ``num_steps < t_end - t_start`` the steps jump multiple indices
        at once (subsampled DDIM, same technique as fast DDIM sampling).
        Using the same ``num_steps`` value in :meth:`ddim_step_mean` during
        decoding preserves the round-trip ODE property.

        The intermediate ``x̂_0`` is **not** clamped — clamping accumulates
        irreversible error across many steps.

        Args:
            x_in:      Input image in DDPM space ``[-1, 1]``, ``[N, C, H, W]``.
                       Must correspond to timestep ``t_start``.
            t_end:     Target timestep (≤ T-1 recommended; see warning at T).
            unet:      Frozen UNet denoiser.
            t_start:   Source timestep (default ``0`` = clean image ``x_0``).
            num_steps: Number of UNet evaluations (ODE steps).
                       ``None`` or ``>= t_end - t_start`` → full resolution.
            log_every: Log progress every this many steps (0 = silent).

        Returns:
            ``x_{t_end}`` in DDPM space ``[-1, 1]``, same shape as *x_in*.
        """
        if t_end <= t_start:
            return x_in.clone()

        if t_end > self.num_timesteps:
            t_end = self.num_timesteps
        if t_end == self.num_timesteps:
            warnings.warn(
                f"ddim_invert: t_end={t_end} equals T={self.num_timesteps}, which is outside the "
                "model's trained timestep range (0 to T-1).  The last forward step uses an "
                "extrapolated boundary condition (ᾱ_T→0) that sets x_T = ε_θ(x_{T-1}, T-1) "
                "— a raw UNet output, not a valid DDPM state.  The DDIM round-trip property "
                "is broken at this boundary.  Use t_end ≤ T-1 for a proper round-trip.",
                UserWarning,
                stacklevel=2,
            )

        # Build evenly-spaced (t_cur, t_next) waypoint pairs.
        # Full resolution when num_steps is None or ≥ the range length.
        total = t_end - t_start
        if num_steps is not None and num_steps < total:
            waypoints = sorted({
                round(t_start + total * i / num_steps) for i in range(num_steps + 1)
            })
        else:
            waypoints = list(range(t_start, t_end + 1))
        pairs = list(zip(waypoints[:-1], waypoints[1:]))

        xt = x_in.clone()
        for step_idx, (t, t_next) in enumerate(pairs):
            with torch.no_grad():
                t_tensor = torch.full(
                    (xt.shape[0],), t, device=xt.device, dtype=torch.long
                )
                eps = unet(xt, t_tensor).sample

            # Predict clean image (not clamped — see docstring)
            a_t = self.sqrt_alphas_cumprod[t].float()
            b_t = self.sqrt_one_minus_alphas_cumprod[t].float()
            x0_hat = (xt - b_t * eps) / a_t.clamp(min=1e-4)

            # DDIM forward step: x_t → x_{t_next}
            if t_next < self.num_timesteps:
                a_next = self.sqrt_alphas_cumprod[t_next].float()
                b_next = self.sqrt_one_minus_alphas_cumprod[t_next].float()
            else:
                # t_next = T: extrapolated boundary (ᾱ_T → 0)
                a_next = torch.zeros(1, device=xt.device, dtype=torch.float32)
                b_next = torch.ones(1, device=xt.device, dtype=torch.float32)
            xt = a_next * x0_hat + b_next * eps

            if log_every > 0 and (step_idx % log_every == 0 or step_idx == len(pairs) - 1):
                logger.debug("DDIM inversion  %d→%d  (%d/%d steps)",
                             t, t_next, step_idx + 1, len(pairs))

        return xt


__all__ = ["load_pretrained_ddpm", "DDPMSchedulerWrapper"]
