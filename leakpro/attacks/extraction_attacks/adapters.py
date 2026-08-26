#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Concrete adapter helpers for arbitrary and OpenAI-style diffusion stacks."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from leakpro.attacks.extraction_attacks.protocols import ConditionGradient
from leakpro.attacks.extraction_attacks.utils import seeded_torch_rng

SampleFunction = Callable[[int, Optional[Sequence[Any]], int], Tensor]
QSampleFunction = Callable[[Tensor, Tensor, Tensor], Tensor]
GuidedSampleFunction = Callable[[int, Tensor, ConditionGradient, int], Tensor]
TimestepFunction = Callable[[Tensor], Tensor]
ConditionEncoder = Callable[[Sequence[Any]], Dict[str, Any]]


@dataclass
class CallableDiffusionAdapter:
    """Bind simple callables to the attack adapter contract."""

    image_shape: tuple[int, int, int]
    sample_fn: SampleFunction
    num_timesteps: int = 0
    q_sample_fn: QSampleFunction | None = None
    guided_sample_fn: GuidedSampleFunction | None = None
    classifier_timestep_fn: TimestepFunction | None = None

    def sample(self, batch_size: int, *, conditions: Sequence[Any] | None, seed: int) -> Tensor:
        """Generate a batch through the configured sampling callable."""
        return self.sample_fn(batch_size, conditions, seed)

    def validate_side_capabilities(self) -> None:
        """Require both forward noising and guided reverse sampling."""
        missing = []
        if not callable(self.sample_fn):
            missing.append("sample_fn")
        if not callable(self.q_sample_fn):
            missing.append("q_sample_fn")
        if not callable(self.guided_sample_fn):
            missing.append("guided_sample_fn")
        if self.classifier_timestep_fn is not None and not callable(self.classifier_timestep_fn):
            missing.append("classifier_timestep_fn")
        if missing:
            raise ValueError(f"SIDE requires callable adapter operations: {', '.join(missing)}.")

    def q_sample(self, clean_images: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        """Apply forward diffusion through the configured callable."""
        if self.q_sample_fn is None:
            raise NotImplementedError("q_sample_fn is required for SIDE.")
        return self.q_sample_fn(clean_images, timesteps, noise)

    def classifier_timesteps(self, timesteps: Tensor) -> Tensor:
        """Map raw timesteps into the classifier's time representation."""
        if self.classifier_timestep_fn is None:
            return timesteps
        return self.classifier_timestep_fn(timesteps)

    def sample_with_classifier_guidance(
        self,
        batch_size: int,
        *,
        labels: Tensor,
        gradient_fn: ConditionGradient,
        seed: int,
    ) -> Tensor:
        """Generate a classifier-guided batch through the configured callable."""
        if self.guided_sample_fn is None:
            raise NotImplementedError("guided_sample_fn is required for SIDE.")

        def classifier_gradient(noisy_images: Tensor, timesteps: Tensor, selected_labels: Tensor) -> Tensor:
            return gradient_fn(noisy_images, self.classifier_timesteps(timesteps), selected_labels)

        return self.guided_sample_fn(batch_size, labels, classifier_gradient, seed)


class OpenAIDiffusionAdapter:
    """Adapter for OpenAI Improved/Guided Diffusion compatible objects.

    Basic sampling supports Improved Diffusion's loop. SIDE additionally
    requires Guided Diffusion's ``cond_fn`` keyword contract.
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
    ) -> None:
        if not hasattr(diffusion, "q_sample") or not hasattr(diffusion, "p_sample_loop"):
            raise TypeError("diffusion must provide q_sample() and p_sample_loop().")
        self.model = model
        self.diffusion = diffusion
        self.image_shape = image_shape
        self.device = torch.device(device)
        self.condition_encoder = condition_encoder
        self.base_model_kwargs = dict(base_model_kwargs or {})
        self.clip_denoised = clip_denoised
        self.num_timesteps = int(diffusion.num_timesteps)
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
        kwargs = self._model_kwargs(conditions)
        with seeded_torch_rng(self.device, seed), torch.no_grad():
            return self.diffusion.p_sample_loop(
                self.model,
                (batch_size, *self.image_shape),
                clip_denoised=self.clip_denoised,
                model_kwargs=kwargs,
                device=self.device,
                progress=False,
            )

    def validate_side_capabilities(self) -> None:
        """Require forward noising and a reverse loop that accepts cond_fn."""
        if not callable(self.diffusion.q_sample) or not callable(self.diffusion.p_sample_loop):
            raise ValueError("SIDE requires callable q_sample() and p_sample_loop() operations.")
        try:
            parameters = inspect.signature(self.diffusion.p_sample_loop).parameters.values()
        except (TypeError, ValueError) as error:
            raise ValueError("SIDE could not inspect p_sample_loop() for the required cond_fn keyword.") from error
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
            raise ValueError("SIDE requires p_sample_loop() to accept a cond_fn keyword or arbitrary keyword arguments.")

    def q_sample(self, clean_images: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        """Apply the wrapped diffusion object's forward process."""
        return self.diffusion.q_sample(clean_images, timesteps, noise=noise)

    def classifier_timesteps(self, timesteps: Tensor) -> Tensor:
        """Match the timestep representation received by the wrapped guidance callback."""
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
        labels = labels.to(self.device)

        def condition(x_t: Tensor, timesteps: Tensor, **_kwargs: object) -> Tensor:
            return gradient_fn(x_t, timesteps, labels)

        with seeded_torch_rng(self.device, seed):
            return self.diffusion.p_sample_loop(
                self.model,
                (batch_size, *self.image_shape),
                clip_denoised=self.clip_denoised,
                cond_fn=condition,
                model_kwargs=dict(self.base_model_kwargs),
                device=self.device,
                progress=False,
            )
