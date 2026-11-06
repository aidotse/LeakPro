#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The reverse-diffusion state x_t, held in the denoiser's [-1, 1] space and decoded to data space."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import Representation

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class DiffusionRepresentation(Representation):
    """Holds the sample ``x_t`` of a reverse-diffusion loop.

    The parameter lives in the denoiser's space [-1, 1], shaped ``[*ctx.axes.shape, *ctx.input_shape]``.
    ``decode()`` maps it to data space, ``to_data(x_t) = (x_t + 1) / 2``, so the stage objective and
    the returned reconstruction see [0, 1] data like every other representation. A fresh bind starts at
    zeros and a bind on a state with an ``"x_t"`` artifact starts from it, which a driver that refines x_t
    (e.g. ``GradientDriver``) continues from. ``DiffusionSamplingDriver`` always replaces the starting
    value with fresh Gaussian noise and writes each reverse step into the parameter.
    """

    def __init__(self) -> None:
        self._param: Tensor | None = None

    @staticmethod
    def to_data(x_t: Tensor) -> Tensor:
        """Denoiser space [-1, 1] to data space [0, 1]."""
        return (x_t + 1.0) / 2.0

    def bind(self, state: State, ctx: Context) -> DiffusionRepresentation:
        """Copy whose x_t starts from the "x_t" artifact, else at zeros, as a leaf requiring grad."""
        bound = copy.copy(self)
        if state.artifacts.get("x_t") is not None:
            start = state.artifacts["x_t"].detach().clone().to(ctx.device)
        else:
            start = torch.zeros((*ctx.axes.shape, *ctx.input_shape), device=ctx.device)
        bound._param = start.requires_grad_(True)
        return bound

    def parameters(self) -> list[dict]:
        """The single "x_t" parameter group."""
        return [{"name": "x_t", "params": [self._param], "lr_scale": 1.0}]

    def decode(self) -> Tensor:
        """Data-space [0, 1] view of the current x_t."""
        return self.to_data(self._param)

    def encode(self, data: Tensor) -> Tensor:
        """Map data-space [0, 1] values into the denoiser space [-1, 1]."""
        return data * 2.0 - 1.0

    def project(self, ctx: Context | None = None) -> None:
        """No-op; the diffusion sampler keeps x_t in range."""
        pass  # the sampler keeps x_t where it belongs; nothing to project

    def to_observable(self) -> Tensor:
        """x_t detached, clamped to [-1, 1] and mapped to data space."""
        return self.to_data(self._param.detach().clamp(-1.0, 1.0))

    def artifacts(self) -> dict[str, Any]:
        """The current x_t, or nothing before binding."""
        return {"x_t": self._param.detach()} if self._param is not None else {}
