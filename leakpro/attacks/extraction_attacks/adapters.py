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
from leakpro.attacks.extraction_attacks.utils_generative import (
    component_metadata,
    json_safe,
    qualified_name,
    seeded_torch_rng,
)
from leakpro.attacks.utils.diff_mi.gaussian_diffusion import GaussianDiffusion
from leakpro.attacks.utils.diff_mi.respace import SpacedDiffusion, space_timesteps
from leakpro.utils.device import get_device, mark_step
from leakpro.utils.logger import logger
from leakpro.utils.save_load import hash_config

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

    def get_extraction_metadata(self, *, guided: bool = False) -> dict[str, Any]:
        """Describe sampler settings for the selected audit mode.

        This covers adapter options and the vendored diffusion schedule.
        Override this getter to add settings specific to a custom process.
        """
        process = self.diffusion if self.sampling_diffusion is None else self.sampling_diffusion
        built_in = type(process) in (GaussianDiffusion, SpacedDiffusion)
        diffusion = {
            "implementation": qualified_name(type(self.diffusion)),
            "num_timesteps": int(process.num_timesteps),
        }
        if built_in:
            timestep_map = getattr(process, "timestep_map", None)
            if timestep_map is None:
                timestep_map = range(process.num_timesteps)
            weight = getattr(process, "w", 1.0)
            diffusion.update(
                timestep_map=json_safe(list(timestep_map)),
                rescale_timesteps=bool(getattr(process, "rescale_timesteps", False)),
                original_num_steps=int(getattr(process, "original_num_steps", self.num_timesteps)),
                classifier_free_guidance_weight=float(weight) if weight is not None else None,
                model_mean_type=process.model_mean_type.name,
                model_var_type=process.model_var_type.name,
                forward_betas_hash=hash_config({"betas": json_safe(self.diffusion.betas)}),
            )
            diffusion["reverse_betas_hash"] = (
                diffusion["forward_betas_hash"] if process is self.diffusion
                else hash_config({"betas": json_safe(process.betas)})
            )
        if guided and self.sampling_diffusion is not None:
            diffusion["guidance_input_rescaled"] = bool(self.diffusion.rescale_timesteps)
        elif guided and not built_in and (
            getattr(self.classifier_timesteps, "__func__", None) is DiffusionAdapter.classifier_timesteps
        ):
            timestep_map = getattr(self.diffusion, "timestep_map", None)
            if timestep_map is not None:
                diffusion["timestep_map"] = json_safe(list(timestep_map))
                diffusion["rescale_timesteps"] = bool(getattr(self.diffusion, "rescale_timesteps", False))
                if diffusion["rescale_timesteps"]:
                    diffusion["original_num_steps"] = int(self.diffusion.original_num_steps)
        return {
            "kind": "ancestral" if self.sampling_diffusion is None else "ddim",
            "device": str(self.device),
            "diffusion": diffusion,
            "clip_denoised": self.clip_denoised,
            "clip_guided": bool(guided and self.sampling_diffusion is not None and self.clip_guided),
            "ddim_eta": None if self.sampling_diffusion is None else 0.0,
            "model_kwargs": _ordered_settings(self.base_model_kwargs),
            "torch_default_dtype": str(torch.get_default_dtype()),
        }

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


def _ordered_settings(settings: dict[str, Any]) -> object:
    """Encode bindings as pairs so sorting JSON object keys cannot change their order."""
    try:
        return json_safe(settings, ordered=True)
    except TypeError as error:
        raise ValueError(f"Invalid sampler metadata. {error}") from error


def _declared_sampler_settings(
    getter: Callable[..., Any], *, guided: bool, optional_mode: bool = False,
) -> dict[str, Any]:
    try:
        settings = getter(guided=guided)
    except NotImplementedError:
        return {}
    except ValueError as error:
        if not optional_mode:
            raise
        logger.warning("Could not read optional unguided sampler settings: %s", error)
        return {}
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise ValueError("get_extraction_metadata() must return a dictionary or None.")
    if settings.keys() == {"target", "sampler"} and all(isinstance(value, dict) for value in settings.values()):
        return {key: _ordered_settings(value) for key, value in settings.items()}
    return {"sampler": _ordered_settings(settings)}


def _adapter_sampler_metadata(adapter: ExtractionAdapter[Tensor], *, guided: bool) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    getter = getattr(adapter, "get_extraction_metadata", None)
    if isinstance(adapter, DiffusionAdapter):
        metadata["sampler"] = DiffusionAdapter.get_extraction_metadata(adapter, guided=guided)
        if getattr(getter, "__func__", None) is DiffusionAdapter.get_extraction_metadata:
            return metadata
    if callable(getter):
        declared = _declared_sampler_settings(getter, guided=guided)
        if declared:
            metadata["declared_settings"] = declared
        if guided:
            synthetic = _declared_sampler_settings(getter, guided=False, optional_mode=True)
            if synthetic:
                metadata["declared_synthetic_settings"] = synthetic
    return metadata


def adapter_metadata(
    adapter: ExtractionAdapter[Tensor], *, guided: bool, conditional: bool,
) -> dict[str, Any]:
    """Record the model, adapter options, and applicable optional sampler settings."""
    shape = getattr(adapter, "image_shape", None)
    if (
        not isinstance(shape, tuple) or len(shape) != 3
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in shape)
    ):
        raise ValueError("adapter.image_shape must be a positive CHW tuple.")

    model = getattr(adapter, "model", None)
    if not isinstance(model, nn.Module):
        raise ValueError("Weight-based extraction result IDs require adapter.model to contain the audited torch model.")
    metadata = {
        "adapter": qualified_name(type(adapter)),
        "target": component_metadata(model),
        "image_shape": list(shape),
        **_adapter_sampler_metadata(adapter, guided=guided),
    }
    if guided and isinstance(timesteps := getattr(adapter, "num_timesteps", None), int):
        metadata["forward_steps"] = timesteps
    if conditional and (encoder := getattr(adapter, "condition_encoder", None)) is not None:
        metadata["condition_encoder"] = component_metadata(encoder)
    return metadata
