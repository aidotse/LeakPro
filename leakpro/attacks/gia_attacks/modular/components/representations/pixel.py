#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Direct pixel representation, optimized in data space or in model-input space."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.initializers import FromState, Uniform
from leakpro.attacks.gia_attacks.modular.core.interfaces import SPACES, Initializer, Modality, Representation, Space

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class PixelRepresentation(Representation):
    """Direct pixel optimization, in data space x or in model-input space z.

    ``space="data"`` optimizes x itself. ``space="model_input"`` optimizes z, the tensor the model
    consumes: ``decode()`` maps z back to data space and ``model_input()`` returns z exactly, so the
    optimizer steps in the same coordinates as an attack written against normalized inputs.

    ``clip`` keeps the iterate in the box ``[min_val, max_val]`` of data space, expressed in the
    parameter's space (for z, the box's image under the modality). ``clip=False`` leaves the
    iterate unconstrained, both in ``project()`` and in ``to_observable()``.

    The starting iterate comes, in order, from ``state.input``, from the state's artifacts
    (``"model_input"`` for a z-space resume, else ``"pixels"``), or from ``init`` (default
    ``Uniform(min_val, max_val)`` in data space). A sample in another space than the parameter's is
    mapped through the modality.
    """

    def __init__(
        self,
        min_val: float = 0.0,
        max_val: float = 1.0,
        *,
        space: Space = "data",
        init: Initializer | None = None,
        clip: bool = True,
    ) -> None:
        if space not in SPACES:
            raise ValueError(f"space must be 'data' or 'model_input', got {space!r}")
        self.min_val = min_val
        self.max_val = max_val
        self.space = space
        self.init = init
        self.clip = clip
        self._param: Tensor | None = None
        self._modality: Modality | None = None
        self._lo: float | Tensor = min_val
        self._hi: float | Tensor = max_val

    def _to_param_space(self, tensor: Tensor, space: Space) -> Tensor:
        if space == self.space:
            return tensor
        if self.space == "model_input":
            return self._modality.to_model_input(tensor)
        return self._modality.strip_model_input(tensor)

    def bind(self, state: State, ctx: Context) -> PixelRepresentation:
        """Copy whose pixels start from state.input, a state artifact, or the initializer, as a leaf requiring grad."""
        bound = copy.copy(self)
        bound._modality = ctx.modality
        if bound.space == "model_input":
            box_shape = (1, *ctx.input_shape[1:])
            bound._lo = ctx.modality.to_model_input(torch.full(box_shape, float(bound.min_val), device=ctx.device))
            bound._hi = ctx.modality.to_model_input(torch.full(box_shape, float(bound.max_val), device=ctx.device))
        else:
            bound._lo, bound._hi = bound.min_val, bound.max_val

        if state.input is not None:
            start = bound._to_param_space(state.input.detach(), "data")
        elif bound.space == "model_input" and "model_input" in state.artifacts:
            start = state.artifacts["model_input"].detach()
        elif "pixels" in state.artifacts:
            start = bound._to_param_space(state.artifacts["pixels"].detach(), "data")
        else:
            init = bound.init if bound.init is not None else Uniform(bound.min_val, bound.max_val, space="data")
            if isinstance(init, FromState):
                raise ValueError("FromState initializer needs a state with an input or pixel artifacts")
            shape = (*ctx.axes.shape, *ctx.input_shape)
            start = bound._to_param_space(init.sample(shape, ctx.device), init.space)
        bound._param = start.clone().requires_grad_(True)
        return bound

    def parameters(self) -> list[dict]:
        """The single pixel parameter group, named "pixels" in data space and "model_input" in model-input space."""
        name = "pixels" if self.space == "data" else "model_input"
        return [{"name": name, "params": [self._param], "lr_scale": 1.0}]

    def decode(self) -> Tensor:
        """The data-space pixels, mapped back from model-input space when optimizing there."""
        if self.space == "data":
            return self._param
        return self._modality.strip_model_input(self._param)

    def model_input(self, modality: Modality) -> Tensor:
        """The model-input tensor: the parameter itself in model-input space, else the pixels mapped through the modality."""
        if self.space == "model_input":
            return self._param
        return modality.to_model_input(self._param)

    def encode(self, data: Tensor) -> Tensor:
        """Clip data to the box when clip is set and map it into the parameter's space."""
        if self.clip:
            data = data.clamp(self.min_val, self.max_val)
        return self._to_param_space(data, "data")

    @torch.no_grad()
    def project(self, ctx: Context | None = None) -> None:  # noqa: ARG002 -- interface signature
        """Clamp the parameter to its box when clip is set."""
        if self.clip:
            self._param.clamp_(self._lo, self._hi)

    def to_observable(self) -> Tensor:
        """The decoded pixels, detached and clamped to [min_val, max_val] when clip is set."""
        x = self.decode().detach()
        return x.clamp(self.min_val, self.max_val) if self.clip else x

    def artifacts(self) -> dict[str, Any]:
        """The pixels, plus the model-input tensor when optimizing in model-input space."""
        if self._param is None:
            return {}
        if self.space == "data":
            return {"pixels": self._param.detach()}
        return {"pixels": self.decode().detach(), "model_input": self._param.detach()}
