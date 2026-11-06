#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Utilities for loading and managing diffusion models for gradient inversion attacks.

This module provides functions to load pre-trained DDPM models via the
``diffusers`` library and exposes lightweight helpers that the
``DiffusionSamplingDriver`` uses during the reverse-process loop.

The design mirrors :mod:`leakpro.fl_utils.gan_handler` — a thin wrapper
that isolates the third-party dependency and keeps attack code clean.

Requires:
    pip install diffusers  (part of the ``federated`` extra: pip install "leakpro[federated]")
"""

from __future__ import annotations

import logging
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
    **kwargs: Any,  # noqa: ANN401 -- genuinely heterogeneous value
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
        from diffusers import DDPMPipeline, DDPMScheduler
    except ImportError as exc:
        raise ImportError(
            "The `diffusers` library is required for diffusion-based GIA "
            "attacks.  Install it with:  pip install diffusers"
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

    Timesteps are 0-indexed: ``ᾱ_t = ∏_{s=0}^{t} α_s``, and ``t_prev = -1`` denotes the clean image
    (``ᾱ_{-1} = 1``). Every reverse-step helper takes the pair ``(t, t_prev)`` with ``t_prev < t``, so
    subsampled sampling, which skips timesteps, uses the posterior of the actual jump.

    Attributes:
        num_timesteps: Total diffusion steps ``T``.
        betas:         β_t schedule  ``[T]``.
        alphas:        α_t = 1 − β_t  ``[T]``.
        alphas_cumprod: ᾱ_t  ``[T]``.
        sqrt_alphas_cumprod:          √ᾱ_t  ``[T]``.
        sqrt_one_minus_alphas_cumprod: √(1 − ᾱ_t)  ``[T]``.
        posterior_variance:  one-step posterior variance β̃_t = β_t (1 − ᾱ_{t−1}) / (1 − ᾱ_t)  ``[T]``.

    """

    def __init__(self, scheduler: Any, device: torch.device) -> None:  # noqa: ANN401 -- genuinely heterogeneous value
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
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device, dtype=torch.float64), alphas_cumprod[:-1]])
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # ------------------------------------------------------------------ #
    # Helpers used by the reverse-diffusion loop                          #
    # ------------------------------------------------------------------ #

    def _alpha_bar(self, t: int) -> torch.Tensor:
        """ᾱ_t, with ᾱ_{-1} = 1 for the clean image."""
        if t < 0:
            return torch.ones((), device=self.alphas_cumprod.device, dtype=torch.float64)
        return self.alphas_cumprod[t]

    def _jump_posterior(self, t: int, t_prev: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Coefficients ``(c_x0, c_xt, variance)`` of q(x_{t_prev} | x_t, x_0) for a reverse jump t → t_prev.

        With the jump's noise level β̂ = 1 − ᾱ_t / ᾱ_{t_prev}:

            mean     = √ᾱ_{t_prev} β̂ / (1 − ᾱ_t) · x_0  +  √(ᾱ_t / ᾱ_{t_prev}) (1 − ᾱ_{t_prev}) / (1 − ᾱ_t) · x_t
            variance = β̂ (1 − ᾱ_{t_prev}) / (1 − ᾱ_t)

        For ``t_prev = t − 1`` (the default) β̂ = β_t and these are the standard DDPM posterior.
        """
        if t_prev is None:
            t_prev = t - 1
        if not t_prev < t:
            raise ValueError(f"reverse step needs t_prev < t, got t={t}, t_prev={t_prev}")
        abar_t, abar_prev = self._alpha_bar(t), self._alpha_bar(t_prev)
        alpha_jump = abar_t / abar_prev
        beta_jump = 1.0 - alpha_jump
        denom = 1.0 - abar_t
        c_x0 = abar_prev.sqrt() * beta_jump / denom
        c_xt = alpha_jump.sqrt() * (1.0 - abar_prev) / denom
        var = beta_jump * (1.0 - abar_prev) / denom
        return c_x0.float(), c_xt.float(), var.float()

    def predict_x0_from_eps(self, xt: torch.Tensor, t: int, eps_pred: torch.Tensor) -> torch.Tensor:
        """Recover x̂_0 from the UNet noise prediction ε_θ(x_t, t).

        x̂_0 = (x_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t
        """
        a = self.sqrt_alphas_cumprod[t].float()
        b = self.sqrt_one_minus_alphas_cumprod[t].float()
        return (xt - b * eps_pred) / a

    def posterior_mean(self, x0_pred: torch.Tensor, xt: torch.Tensor, t: int, t_prev: int | None = None) -> torch.Tensor:
        """Posterior mean of x_{t_prev} given x_t and x̂_0 (``t_prev`` defaults to ``t − 1``)."""
        c_x0, c_xt, _ = self._jump_posterior(t, t_prev)
        return c_x0 * x0_pred + c_xt * xt

    def posterior_std(self, t: int, t_prev: int | None = None) -> torch.Tensor:
        """Posterior standard deviation of x_{t_prev} given x_t and x_0 (``t_prev`` defaults to ``t − 1``)."""
        return self._jump_posterior(t, t_prev)[2].clamp(min=0.0).sqrt()

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

        Both formulas are linear in *x0_pred*; swapping ``x0_pred`` for an optimised ``x̂_0*``
        produces the correct adjusted mean for either sampler.

        DDPM:  the posterior mean of x_{t_prev} given (x_t, x̂_0), for a jump of any length
        DDIM:  x_{t_prev} = √ᾱ_{t_prev} · x̂_0  +  √(1−ᾱ_{t_prev}) · ε_θ

        Args:
            x0_pred:  Clean-image estimate x̂_0  ``[B, C, H, W]``.
            xt:       Current noisy image x_t    ``[B, C, H, W]``
                      (used only for DDPM; ignored for DDIM).
            eps_pred: UNet noise prediction ε_θ  ``[B, C, H, W]``
                      (used only for DDIM; ignored for DDPM).
            t:        Current timestep.
            t_prev:   Target timestep (must be < *t*).  ``-1`` is the clean image: both samplers then
                      return *x0_pred*.
            sampler:  ``"ddpm"`` (stochastic ancestral) or
                      ``"ddim"`` (deterministic η=0).

        Returns:
            Step-mean tensor, same shape as *x0_pred*.

        """
        if sampler == "ddim":
            return self.ddim_step_mean(x0_pred, eps_pred, t, t_prev=t_prev)
        return self.posterior_mean(x0_pred, xt, t, t_prev)


__all__ = ["load_pretrained_ddpm", "DDPMSchedulerWrapper"]
