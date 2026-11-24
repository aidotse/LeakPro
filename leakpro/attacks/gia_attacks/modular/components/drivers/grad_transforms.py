#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Gradient transforms ``GradientDriver`` applies between the backward pass and the optimizer step."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import GradTransform


@dataclass(frozen=True)
class SignGradient(GradTransform):
    """Replace every gradient by its elementwise sign (signed gradient descent)."""

    @torch.no_grad()
    def __call__(self, params: Sequence[Tensor]) -> None:
        """Set ``p.grad = sign(p.grad)`` for every parameter that has a gradient."""
        for p in params:
            if p.grad is not None:
                p.grad.copy_(torch.sign(p.grad))


@dataclass(frozen=True)
class GaussianNoise(GradTransform):
    """Add zero-mean Gaussian noise of standard deviation ``std`` to every gradient.

    With ``seed`` set, ``bind`` gives each run its own generator, so the global RNG is untouched and
    the noise is reproducible; without it the noise is drawn from the global RNG. ``std == 0`` is a
    no-op that draws nothing.
    """

    std: float
    seed: int | None = None

    def __post_init__(self) -> None:
        """Rejects a negative noise standard deviation."""
        if self.std < 0:
            raise ValueError(f"std must be non-negative, got {self.std}")

    def bind(self, device: torch.device) -> GradTransform:
        """Returns a copy of the transform that owns a seeded generator on the given device."""
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=device).manual_seed(self.seed)
        return _BoundGaussianNoise(self.std, generator)

    def __call__(self, params: Sequence[Tensor]) -> None:  # noqa: ARG002 -- interface signature
        """Adds nothing when std is zero and otherwise fails because the transform is unbound."""
        if self.std == 0:
            return
        raise RuntimeError("bind() the transform first")


@dataclass(frozen=True, eq=False)
class _BoundGaussianNoise(GradTransform):
    """``GaussianNoise`` of one run, owning that run's generator."""

    std: float
    generator: torch.Generator | None

    @torch.no_grad()
    def __call__(self, params: Sequence[Tensor]) -> None:
        if self.std == 0:
            return
        for p in params:
            if p.grad is None:
                continue
            if self.generator is None:
                noise = torch.randn_like(p.grad)
            else:
                noise = torch.randn(p.grad.shape, generator=self.generator, device=p.grad.device, dtype=p.grad.dtype)
            p.grad.add_(noise * self.std)


@dataclass(frozen=True)
class ClipGradNorm(GradTransform):
    """Scale all gradients together so their joint ``norm_type``-norm is at most ``max_norm``."""

    max_norm: float
    norm_type: float = 2.0

    @torch.no_grad()
    def __call__(self, params: Sequence[Tensor]) -> None:
        """Clips the global gradient norm of the parameters that have a gradient."""
        with_grad = [p for p in params if p.grad is not None]
        if with_grad:
            torch.nn.utils.clip_grad_norm_(with_grad, self.max_norm, norm_type=self.norm_type)
