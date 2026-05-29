#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Transition strategies for sequential multi-stage optimization.

This module defines strategies for transforming state between optimization stages.
Useful for progressive attacks like GIFD where each stage optimizes in different
parameter spaces but needs to pass information forward.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.config.registry import register
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    Component,
    ComponentMetadata,
)

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import RunContext, WorkingState

logger = logging.getLogger(__name__)


class TransitionStrategy(Component):
    """Base class for stage transition strategies.

    Defines how to transform state between optimization stages.
    Each transition receives the previous stage's WorkingState and
    produces an updated state for the next stage.
    """

    @abstractmethod
    def apply(
        self,
        state: "WorkingState",
        ctx: "RunContext",
    ) -> "WorkingState":
        """Transform previous stage output into next stage input.

        Reads state.reconstruction and state.optimizable_tensor.
        Sets state.reconstruction to the new input and clears
        state.optimizable_tensor (the next stage starts fresh).

        Args:
            state: WorkingState from the completed stage.
            ctx: Immutable run context (model, observations, etc.).

        Returns:
            Updated WorkingState ready for the next stage.
        """
        pass


@register("transition.reconstruction")
class ReconstructionTransition(TransitionStrategy):
    """Transition using previous stage's reconstruction.

    Passes the reconstruction from the previous stage to initialize the next stage.
    The next stage will use this reconstruction to initialize its own parameters:
    - If next stage has no representation: directly optimize the reconstruction
    - If next stage has a representation: initialize parameters in its space from reconstruction

    This is the standard transition used in most multi-stage attacks like GIAS:
    - Stage 1 (latent space): Outputs reconstruction in data space
    - Stage 2 (pixel space): Uses that reconstruction to initialize pixel optimization
    """

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for this component."""
        return ComponentMetadata(
            name="reconstruction_transition",
            required_capabilities={},
        )

    def apply(
        self,
        state: "WorkingState",
        ctx: "RunContext",
    ) -> "WorkingState":
        """Pass previous reconstruction unchanged to the next stage."""
        logger.debug("Reconstruction transition: passing reconstruction to next stage")
        state.optimizable_tensor = None
        return state


@register("transition.latent_code")
class LatentCodeTransition(TransitionStrategy):
    """Transition that passes the optimised latent codes (not the pixel reconstruction).

    Use this between a frozen-generator stage (GANRepresentation) and an
    unfrozen-generator stage (UnfrozenGANRepresentation) so that stage 2
    starts from the best latent codes found in stage 1 rather than from the
    corresponding pixel-space image.

    The latent tensor from stage 1 is stored as `prev_optimizable` with shape
    ``[N, latent_dim]``.  This transition reshapes it to
    ``[1, N, 1, latent_dim]`` (adding the E and G batch dimensions expected by
    the representation strategy) before handing it to the next stage.

    If no optimizable tensor is available (e.g. stage 1 used pixel space and
    there are no latent codes), the method falls back to passing the
    pixel-space reconstruction unchanged.
    """

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for this component."""
        return ComponentMetadata(
            name="latent_code_transition",
            required_capabilities={},
        )

    def apply(
        self,
        state: "WorkingState",
        ctx: "RunContext",
    ) -> "WorkingState":
        """Pass reshaped latent codes as initialisation for the next stage.

        If no optimizable tensor is available, falls back to pixel-space reconstruction.
        """
        prev_optimizable = state.optimizable_tensor
        if prev_optimizable is None:
            logger.warning(
                "LatentCodeTransition: no optimizable tensor from previous stage — "
                "falling back to pixel-space reconstruction."
            )
            state.optimizable_tensor = None
            return state

        # prev_optimizable: [N, latent_dim]  →  [1, N, 1, latent_dim]
        latent = prev_optimizable.unsqueeze(0).unsqueeze(2)
        logger.debug(f"Latent code transition: {prev_optimizable.shape} → {latent.shape}")
        state.reconstruction = latent
        state.optimizable_tensor = None
        return state


@register("transition.gifd_layer")
class GIFDLayerTransition(TransitionStrategy):
    """GIFD transition: apply one generator layer to advance to the next feature domain.

    Given the best intermediate features ``h*_k`` from stage *k*, this
    transition computes the initial features for stage *k+1* by applying
    generator layer *k*:

    .. math::

        h^0_{k+1} = G_k(h^*_k)

    For the first transition (latent space → first feature domain) set
    ``from_latent=True``.  The latent codes ``z*`` are then reshaped to
    ``[batch, latent_dim, 1, 1]`` before layer 0 is applied.

    Args:
        generator: A :class:`~leakpro.fl_utils.gan_models.LayeredGANWrapper`
                   that exposes :meth:`apply_layer`.
        layer_idx: Index of the generator layer group to apply (0-indexed).
                   This is *k* in Algorithm 1 of the GIFD paper.
        from_latent: If ``True`` the previous stage's optimisable tensor is
                     1-D latent codes ``[N, latent_dim]`` that need to be
                     reshaped to ``[N, latent_dim, 1, 1]`` before applying
                     layer 0.  Set to ``True`` only for the transition between
                     stage 0 (latent search) and stage 1.
    """

    def __init__(
        self,
        generator: nn.Module,
        layer_idx: int,
        from_latent: bool = False,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.layer_idx = layer_idx
        self.from_latent = from_latent

        # Freeze generator during transitions
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata."""
        return ComponentMetadata(
            name="gifd_layer_transition",
            required_capabilities={"has_pretrained_gan": True},
        )

    def apply(
        self,
        state: "WorkingState",
        ctx: "RunContext",
    ) -> "WorkingState":
        """Apply one generator layer to produce initial features for the next stage."""
        prev_optimizable = state.optimizable_tensor
        if prev_optimizable is None:
            logger.warning(
                "GIFDLayerTransition: no optimisable tensor from previous stage — "
                "falling back to pixel-space reconstruction."
            )
            state.optimizable_tensor = None
            return state

        with torch.no_grad():
            features = prev_optimizable  # [N, ...] or [E, N, G, ...] or [E, N, G, latent_dim]

            # Strip E and G batch dims if present (composable_optimizer stores
            # tensors as [E, N, G, ...]; transitions need [N, ...]).
            if features.ndim == 6:
                # [E, N, G, C, H, W] → [N, C, H, W] (E=1, G=1 expected)
                features = features[0, :, 0]
            elif features.ndim == 4 and not self.from_latent:
                # [N, C, H, W] already
                pass
            elif features.ndim == 4 and self.from_latent:
                # [E, N, G, latent_dim] → [N, latent_dim]
                features = features[0, :, 0]

            if self.from_latent:
                # Latent codes need spatial dims: [N, latent_dim] → [N, latent_dim, 1, 1]
                if features.ndim == 2:
                    features = features.unsqueeze(-1).unsqueeze(-1)

            # Apply exactly one layer group: [N, C_in, H_in, W_in] → [N, C_out, H_out, W_out]
            next_features = self.generator.apply_layer(features, self.layer_idx)

        # Add E=1 and G=1 batch dims: [N, C, H, W] → [1, N, 1, C, H, W]
        result = next_features.unsqueeze(0).unsqueeze(2)

        logger.debug(
            f"GIFDLayerTransition (layer {self.layer_idx}, from_latent={self.from_latent}): "
            f"{prev_optimizable.shape} → {result.shape}"
        )
        state.reconstruction = result
        state.optimizable_tensor = None
        return state


__all__ = [
    "TransitionStrategy",
    "ReconstructionTransition",
    "LatentCodeTransition",
    "GIFDLayerTransition",
]
