#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Model and sampler boundary for diffusion extraction attacks."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

from leakpro.attacks.extraction_attacks.protocols import ConditionGradient, ExtractionAdapter
from leakpro.attacks.extraction_attacks.utils_generative import seeded_torch_rng
from leakpro.attacks.utils.diff_mi.gaussian_diffusion import GaussianDiffusion
from leakpro.attacks.utils.diff_mi.respace import SpacedDiffusion, space_timesteps
from leakpro.utils.device import get_device, mark_step

ConditionEncoder = Callable[[Sequence[Any]], dict[str, Any]]


class DiffusionAdapter(ExtractionAdapter[Tensor]):
    """Adapter for OpenAI Improved/Guided Diffusion compatible objects.

    Basic sampling supports Improved Diffusion's loop. SIDE additionally
    requires Guided Diffusion's ``cond_fn`` keyword contract. Supplying
    ``sampling_steps`` requires LeakPro's vendored diffusion process and selects
    its eta-zero DDIM sampler, with native indices for SIDE's classifier. ``clip_guided`` projects guided
    clean predictions to [-1, 1] to stabilize the terminal cosine step.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        diffusion: Any,  # noqa: ANN401 - dynamic third-party diffusion API boundary
        image_shape: tuple[int, int, int],
        device: str = "cpu",
        condition_encoder: ConditionEncoder | None = None,
        base_model_kwargs: dict[str, Any] | None = None,
        clip_denoised: bool = True,
        sampling_steps: int | None = None,
        clip_guided: bool = False,
    ) -> None:
        if not hasattr(diffusion, "q_sample") or not hasattr(diffusion, "p_sample_loop"):
            raise TypeError("diffusion must provide q_sample() and p_sample_loop().")
        self.model = model
        self.diffusion = diffusion
        self.image_shape = image_shape
        self.device = get_device() if device == "auto" else torch.device(device)
        self.condition_encoder = condition_encoder
        self.base_model_kwargs = dict(base_model_kwargs or {})
        self.clip_denoised = clip_denoised
        self.num_timesteps = int(diffusion.num_timesteps)
        self.sampling_diffusion: SpacedDiffusion | None = None
        self.clip_guided = clip_guided
        if sampling_steps is not None:
            if not isinstance(diffusion, GaussianDiffusion):
                raise TypeError("sampling_steps requires LeakPro's vendored GaussianDiffusion process.")
            if not 2 <= sampling_steps <= self.num_timesteps:
                raise ValueError("sampling_steps must be between 2 and diffusion.num_timesteps.")
            if getattr(diffusion, "original_num_steps", self.num_timesteps) != self.num_timesteps:
                raise ValueError("DDIM respacing requires the full training diffusion process.")
            self.sampling_diffusion = SpacedDiffusion(
                use_timesteps=space_timesteps(self.num_timesteps, [sampling_steps]),
                betas=diffusion.betas,
                model_mean_type=diffusion.model_mean_type,
                model_var_type=diffusion.model_var_type,
                loss_type=diffusion.loss_type,
                rescale_timesteps=diffusion.rescale_timesteps,
                w=getattr(diffusion, "w", 1.0),
            )
        self.model.to(self.device).eval()

    def _model_kwargs(self, conditions: Sequence[Any] | None) -> dict[str, Any]:
        kwargs = dict(self.base_model_kwargs)
        if conditions is not None:
            if self.condition_encoder is None:
                raise ValueError("conditions were supplied but no condition_encoder was configured.")
            encoded = self.condition_encoder(conditions)
            overlap = set(kwargs).intersection(encoded)
            if overlap:
                raise ValueError(f"condition_encoder overwrote base model kwargs: {sorted(overlap)}")
            kwargs.update(encoded)
        return kwargs

    def sample(self, batch_size: int, *, conditions: Sequence[Any] | None, seed: int) -> Tensor:
        """Generate a batch with an OpenAI-style reverse diffusion loop."""
        return self._sample(batch_size, self._model_kwargs(conditions), seed)

    def _sample(
        self,
        batch_size: int,
        model_kwargs: dict[str, Any],
        seed: int,
        cond_fn: Callable[..., Tensor] | None = None,
    ) -> Tensor:
        """Select the reverse process while restoring the caller's RNG state."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        shape = (batch_size, *self.image_shape)
        guidance = {} if cond_fn is None else {"cond_fn": cond_fn}
        with seeded_torch_rng(self.device, seed), torch.no_grad():
            if self.sampling_diffusion is None:
                return self.diffusion.p_sample_loop(
                    self.model, shape, clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs, device=self.device, progress=False, **guidance,
                )
            generator = torch.Generator(device="cpu").manual_seed(seed)
            images = torch.randn(shape, generator=generator, device="cpu").to(self.device)
            for output in self.sampling_diffusion.ddim_sample_loop_progressive(
                self.model, shape, noise=images, clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs, device=self.device, progress=False,
                clip_guided=self.clip_guided, **guidance,
            ):
                images = output["sample"]
                mark_step(self.device)
            return images.clamp(-1, 1).detach()

    def validate_side_capabilities(self) -> None:
        """Require forward noising and a reverse loop that accepts cond_fn."""
        loop = (self.diffusion.p_sample_loop if self.sampling_diffusion is None
                else self.sampling_diffusion.ddim_sample_loop_progressive)
        if not callable(self.diffusion.q_sample) or not callable(loop):
            raise ValueError("SIDE requires callable q_sample() and reverse sampling operations.")
        try:
            parameters = inspect.signature(loop).parameters.values()
        except (TypeError, ValueError) as error:
            raise ValueError("SIDE could not inspect the reverse sampling loop for the required cond_fn keyword.") from error
        parameters = tuple(parameters)
        positional_only_cond_fn = any(
            parameter.name == "cond_fn" and parameter.kind is inspect.Parameter.POSITIONAL_ONLY
            for parameter in parameters
        )
        accepts_explicit_cond_fn = any(
            parameter.name == "cond_fn"
            and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            for parameter in parameters
        )
        accepts_cond_fn = accepts_explicit_cond_fn or (
            not positional_only_cond_fn
            and any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        )
        if not accepts_cond_fn:
            raise ValueError("SIDE requires the reverse sampling loop to accept cond_fn or arbitrary keyword arguments.")

    def q_sample(self, clean_images: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        """Apply the wrapped diffusion object's forward process."""
        return self.diffusion.q_sample(
            clean_images.to(self.device), timesteps.to(self.device), noise=noise.to(self.device),
        )

    def classifier_timesteps(self, timesteps: Tensor) -> Tensor:
        """Match the timestep representation received by the wrapped guidance callback."""
        if self.sampling_diffusion is not None:
            return timesteps
        timestep_map = getattr(self.diffusion, "timestep_map", None)
        if timestep_map is not None:
            mapped = torch.as_tensor(timestep_map, device=timesteps.device, dtype=timesteps.dtype)[timesteps]
            if bool(getattr(self.diffusion, "rescale_timesteps", False)):
                original_steps = int(self.diffusion.original_num_steps)
                mapped = mapped.float() * (1_000.0 / original_steps)
            return mapped
        scale = getattr(self.diffusion, "_scale_timesteps", None)
        return scale(timesteps) if callable(scale) else timesteps

    def sample_with_classifier_guidance(
        self,
        batch_size: int,
        *,
        labels: Tensor,
        gradient_fn: ConditionGradient,
        seed: int,
    ) -> Tensor:
        """Generate with classifier gradients supplied as ``cond_fn``."""
        if labels.shape != (batch_size,):
            raise ValueError("Classifier-guided sampling requires one label per image.")
        labels = labels.to(self.device)

        def condition(x_t: Tensor, timesteps: Tensor, **_kwargs: object) -> Tensor:
            if self.sampling_diffusion is not None:
                # The U-Net uses rescaled times; the example classifier uses native indices.
                if self.diffusion.rescale_timesteps:
                    timesteps = (timesteps * (self.num_timesteps / 1_000.0)).round()
                timesteps = timesteps.long()
            gradient = gradient_fn(x_t, timesteps, labels).detach()
            if gradient.shape != x_t.shape or not torch.isfinite(gradient).all():
                raise ValueError("The guidance gradient must match the images and contain finite values.")
            return gradient

        return self._sample(batch_size, dict(self.base_model_kwargs), seed, condition)
