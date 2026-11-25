#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Generator wrapper that lets GIFD run a GAN generator layer by layer."""

from __future__ import annotations

import torch
from torch import nn


class LayeredGANWrapper(nn.Module):
    """Generator wrapper enabling layer-by-layer execution for GIFD.

    Splits a generator's ``main`` Sequential into ConvTranspose2d-led blocks
    (one block per semantic "level"), and exposes:

    * :meth:`forward_from` – run from a specific layer index to the output
    * :meth:`apply_layer` – run exactly one layer block (used when moving to the next feature level)
    * :meth:`forward` – standard end-to-end forward pass (drop-in replacement)

    The wrapper is compatible with :class:`GANRepresentation` for latent-space
    stage 0 and with :class:`GIFDIntermediateRepresentation` for intermediate
    feature stages 1 … K.

    Args:
        generator: Base generator with a ``main: nn.Sequential`` attribute
                   (e.g. a DCGAN ``nn.Module`` with a ``main`` Sequential).

    Raises:
        ValueError: If the generator lacks a ``main`` Sequential attribute.

    """

    def __init__(self, generator: nn.Module) -> None:
        """Wrap *generator* and build layer groups."""
        super().__init__()
        self.generator = generator
        self.latent_dim: int | None = getattr(generator, "latent_dim", None)
        self._build_layer_groups()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_layer_groups(self) -> None:
        """Split the generator's Sequential layers into ConvTranspose2d-led blocks.

        Supports two common layouts:

        * ``generator.main`` – the generator has a ``main: nn.Sequential``
          attribute (e.g. a DCGAN ``nn.Module`` with a ``main`` Sequential).
        * ``generator`` itself is an ``nn.Sequential`` – the generator returned
          by :func:`~leakpro.fl_utils.gan_handler._build_dcgan_generator` is a
          bare ``nn.Sequential`` (used by the DCGAN loaders).
        """
        if isinstance(self.generator, nn.Sequential):
            sequential = self.generator
        elif hasattr(self.generator, "main") and isinstance(self.generator.main, nn.Sequential):
            sequential = self.generator.main
        else:
            raise ValueError(
                "LayeredGANWrapper requires either a bare nn.Sequential generator "
                "or a generator with a 'main: nn.Sequential' attribute "
                "(the DCGAN returned by load_pretrained_gan is compatible)."
            )

        groups: list[nn.Module] = []
        current: list[nn.Module] = []

        for layer in sequential.children():
            if isinstance(layer, nn.ConvTranspose2d) and current:
                # Flush the current group and start a new one
                groups.append(nn.Sequential(*current))
                current = [layer]
            else:
                current.append(layer)

        if current:  # Flush the final group
            groups.append(nn.Sequential(*current))

        self.layer_groups = nn.ModuleList(groups)
        self.num_layer_groups: int = len(self.layer_groups)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002 -- interface signature
        """Standard end-to-end forward pass (all layer groups).

        Args:
            z: Latent codes ``[batch, latent_dim]`` or ``[batch, latent_dim, 1, 1]``
            labels: Ignored (unconditional wrapper).

        Returns:
            Generated images ``[batch, C, H, W]``.

        """
        return self.forward_from(z, start_layer=0)

    def forward_from(self, features: torch.Tensor, start_layer: int) -> torch.Tensor:
        """Run generator from *start_layer* onward.

        Args:
            features: Input tensor.

                * ``start_layer == 0``: latent codes
                  ``[batch, latent_dim]`` or ``[batch, latent_dim, 1, 1]``
                * ``start_layer > 0``: intermediate features
                  ``[batch, C, H, W]``

            start_layer: Index of the first layer group to apply.

        Returns:
            Output tensor (pixel space for the last layer group).

        """
        if start_layer == 0 and features.ndim == 2:
            # Latent codes: add spatial dims for ConvTranspose2d
            features = features.unsqueeze(-1).unsqueeze(-1)

        x = features
        for i in range(start_layer, self.num_layer_groups):
            x = self.layer_groups[i](x)
        return x

    def apply_layer(self, features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply exactly one layer group (used by :class:`GIFDIntermediateRepresentation` to move to the next feature level).

        Args:
            features: Input features ``[batch, C, H, W]``
                      (or ``[batch, latent_dim, 1, 1]`` for layer 0).
            layer_idx: Index of the layer group to apply.

        Returns:
            Output features ``[batch, C', H', W']``.

        """
        return self.layer_groups[layer_idx](features)


__all__ = ["LayeredGANWrapper"]
