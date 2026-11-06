#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GIFD's progressive feature-domain representation (Fang et al. 2023).

Each stage seeds its features from the previous stage's best result inside ``bind``, so a GIFD
attack needs nothing beyond its stages.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.constraints.l1_ball import project_l1_ball
from leakpro.attacks.gia_attacks.modular.components.representations.gan import (
    _check_range,
    _frozen_run_copy,
    to_unit_range,
)
from leakpro.attacks.gia_attacks.modular.core.interfaces import Representation

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State
    from leakpro.fl_utils.gan_models import LayeredGANWrapper


class GIFDIntermediateRepresentation(Representation):
    """Optimizes an intermediate feature tensor h_k, decoding via G_{k..K}(h_k).

    ``decode()`` returns data space ([0, 1] for images): the generator's raw output, spanning
    ``output_range``, is mapped linearly onto [0, 1], exactly as ``GANRepresentation`` does for
    stage 0, so every GIFD stage hands over the same space.

    Args:
        generator: A ``LayeredGANWrapper``
            exposing ``forward_from(features, start_layer)``.
        start_layer: Index of the first layer group ``decode()`` runs from.
        l1_radius: L1-ball radius ``project()`` keeps this stage's feature
            *deviation from its initial (bind-time) value* within — matching
            the per-stage shrinking radius, which constrains how far
            optimization may move away from the seed features, not the
            absolute magnitude of the features themselves. Progressively
            shrunk stage-to-stage by the ``gifd()`` preset, matching
            the paper's shrinking-constraint schedule.
        output_range: ``(lo, hi)`` range of the generator's raw output; ``(-1, 1)`` for a Tanh head.

    """

    def __init__(
        self,
        generator: LayeredGANWrapper,
        start_layer: int,
        l1_radius: float,
        *,
        output_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.generator = generator
        self.output_range = _check_range(output_range)
        self.start_layer = start_layer
        self.l1_radius = l1_radius
        self._features: Tensor | None = None
        self._initial_features: Tensor | None = None
        self._prefix_ndim = 0

    def bind(self, state: State, ctx: Context) -> GIFDIntermediateRepresentation:
        """Start from the features the previous stage handed over, advanced to this stage's layer.

        ``state.artifacts`` is read as follows (``k = start_layer``):

        - ``"start_layer" == k``: this stage's own features (a resumed run); used as they are.
        - ``"start_layer" == k - 1``: the previous GIFD stage's best features ``h_{k-1}``; this
          stage starts from ``G_{k-1}(h_{k-1})``.
        - ``k == 1`` and a ``"latent"``: stage 0's best latent ``z``; this stage starts from
          ``G_0(z)``.

        Raises:
            ValueError: If the artifacts come from none of these predecessors.

        """
        bound = copy.copy(self)
        bound.generator = _frozen_run_copy(self.generator, ctx)
        bound._prefix_ndim = len(ctx.axes.shape)
        artifacts = state.artifacts
        k = self.start_layer
        previous = artifacts.get("start_layer")
        if "features" in artifacts and previous == k:
            features = artifacts["features"].to(ctx.device)
        elif "features" in artifacts and previous == k - 1:
            features = bound._advance(artifacts["features"].to(ctx.device), k - 1)
        elif k == 1 and "latent" in artifacts:
            features = bound._advance(artifacts["latent"].to(ctx.device)[..., None, None], 0)
        else:
            expected = "a GAN stage's 'latent'" if k == 1 else f"'features' with 'start_layer' {k - 1}"
            raise ValueError(
                f"GIFDIntermediateRepresentation(start_layer={k}).bind() needs the artifacts of the previous "
                f"stage ({expected}) or of this stage ('features' with 'start_layer' {k}); "
                f"got keys {sorted(artifacts)} with start_layer={previous!r}."
            )
        bound._features = features.detach().clone().requires_grad_(True)
        bound._initial_features = bound._features.detach().clone()
        return bound

    def _advance(self, features: Tensor, layer_idx: int) -> Tensor:
        """Apply generator layer group ``layer_idx`` to ``[*axes, N, ...]`` features, keeping the prefix."""
        prefix = features.shape[: self._prefix_ndim + 1]
        with torch.no_grad():
            out = self.generator.apply_layer(self._flat(features), layer_idx)
        return out.reshape(*prefix, *out.shape[1:])

    def parameters(self) -> list[dict]:
        """The single "features" parameter group."""
        return [{"name": "features", "params": [self._features], "lr_scale": 1.0}]

    def _flat(self, features: Tensor) -> Tensor:
        """Fold the axis prefix and the batch into one leading dimension, as the generator expects."""
        return features.reshape(-1, *features.shape[self._prefix_ndim + 1:])

    def decode(self) -> Tensor:
        """Run the features through the generator from start_layer and map its output range onto [0, 1].

        Features are ``[*ctx.axes.shape, N, C, H, W]``; the generator sees a flat batch.
        """
        prefix = self._features.shape[: self._prefix_ndim + 1]
        out = self.generator.forward_from(self._flat(self._features), start_layer=self.start_layer)
        out = to_unit_range(out, self.output_range)
        return out.reshape(*prefix, *out.shape[1:])

    def encode(self, data: Tensor) -> Tensor:  # noqa: ARG002 -- Representation interface
        """Not supported: inverting the generator would need its own optimisation."""
        raise NotImplementedError("GAN intermediate features cannot be encoded from data; start from stage artifacts")

    def project(self, ctx: Context | None = None) -> None:  # noqa: ARG002 -- Representation interface; this projection needs no context
        """Project the deviation from the initial features onto the L1 ball of radius l1_radius."""
        with torch.no_grad():
            deviation = self._flat(self._features.data - self._initial_features)
            projected_deviation = project_l1_ball(deviation, self.l1_radius).view_as(self._features)
            self._features.data.copy_(self._initial_features + projected_deviation)

    def to_observable(self) -> Tensor:
        """The decoded batch without gradient tracking."""
        with torch.no_grad():
            return self.decode().detach()

    def artifacts(self) -> dict[str, Any]:
        """The current features and this stage's start_layer, or nothing before binding."""
        if self._features is None:
            return {}
        return {"features": self._features.detach(), "start_layer": self.start_layer}

