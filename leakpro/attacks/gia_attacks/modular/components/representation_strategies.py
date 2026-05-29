#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Representation strategies for transforming optimization parameters to pixel space.

This module provides strategies for mapping between optimization space and
pixel space. This enables attacks that optimize in latent/generative spaces
instead of directly in pixel space.

Strategies:
    - PixelRepresentation: Direct pixel optimization (identity transform)
    - GANRepresentation: Latent → pixels via generator
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.core.component_base import (
    Component,
    ComponentMetadata,
)

logger = logging.getLogger(__name__)


class RepresentationStrategy(Component):
    """Base class for representation strategies.

    Representation strategies transform optimization parameters to pixel space.
    This abstraction allows optimizing in different spaces (pixels, latent, etc.)
    while keeping losses and constraints operating on pixel-space reconstructions.
    """

    @abstractmethod
    def forward(
        self,
        params: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Transform parameters to pixel-space reconstruction.

        Args:
            params: Optimization parameters [E, N, G, *param_shape]
                   where E=epochs, N=batch, G=seeds
            labels: Optional labels for conditional generation

        Returns:
            reconstruction: Pixel-space tensor [E, N, G, C, H, W]
        """
        pass

    @abstractmethod
    def get_parameter_shape(
        self,
        target_pixel_shape: tuple[int, ...],
        **kwargs: Any,
    ) -> tuple[int, ...]:
        """Get parameter shape needed to produce target pixel shape.

        Args:
            target_pixel_shape: Desired output shape [E, N, G, C, H, W]

        Returns:
            param_shape: Shape for optimization parameters
        """
        pass

    def prepare_for_stage(
        self,
        data_mean: torch.Tensor | None,
        data_std: torch.Tensor | None,
    ) -> None:
        """Set dataset normalization parameters (no-op by default).

        Subclasses that produce outputs in a different value range (e.g.
        GAN representations whose generator uses Tanh) should override
        this to store mean/std and apply the conversion inside forward().

        Args:
            data_mean: Per-channel mean [C, 1, 1]
            data_std: Per-channel std  [C, 1, 1]
        """
        pass


class GANRepresentation(RepresentationStrategy):
    """Latent-space optimization with GAN generator.

    Optimizes in the GAN latent space and decodes to pixels via the generator.
    This enforces a strong prior that reconstructions lie on the GAN manifold.

    Args:
        generator: Pre-trained GAN generator model
        latent_dim: Dimension of latent code
        conditional: Whether generator is class-conditional
        num_classes: Number of classes (for conditional GANs)
    """

    def __init__(
        self,
        generator: nn.Module,
        latent_dim: int = 512,
        conditional: bool = False,
        num_classes: int | None = None,
        data_mean: torch.Tensor | None = None,
        data_std: torch.Tensor | None = None,
    ) -> None:
        """Initialize GAN representation.

        Args:
            generator: Pre-trained generator (should be in eval mode)
            latent_dim: Dimension of latent code
            conditional: Whether generator accepts class labels
            num_classes: Number of classes (required if conditional=True)
            data_mean: Per-channel mean [C, 1, 1] for converting GAN output
                       from [-1, 1] to dataset-normalized space.  When
                       provided together with data_std the forward() pass
                       applies: x → (x+1)/2 → (x - mean) / std
            data_std: Per-channel std [C, 1, 1] (see data_mean)
        """
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes
        self.data_mean = data_mean
        self.data_std = data_std

        # NOTE: do NOT eagerly call generator.eval()/freeze here — the generator may be
        # shared with a later unfrozen stage (e.g. GIAS stage 2 UnfrozenGANRepresentation).
        # Applying eval+freeze in __init__ would be overridden by the other stage's __init__
        # and vice-versa, causing undefined ordering bugs.  Instead, each representation
        # applies its mode lazily in prepare_for_stage(), called right before its stage runs.

        if conditional and num_classes is None:
            raise ValueError("num_classes must be provided for conditional GANs")

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for this component."""
        return ComponentMetadata(
            name="gan_representation",
            required_capabilities={
                "has_pretrained_gan": True,
            },
        )

    def forward(
        self,
        params: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode latent codes to pixels via generator.

        Args:
            params: Latent codes [E, N, G, latent_dim]
            labels: Class labels [E, N] (required if conditional=True)

        Returns:
            reconstruction: Generated images [E, N, G, C, H, W]
        """
        if params.ndim != 4:
            raise ValueError(
                f"GANRepresentation expects latent codes with shape [E, N, G, latent_dim] (4D), "
                f"but got tensor with shape {params.shape} ({params.ndim}D). "
                f"This usually means pixel-space data was passed instead of latent codes."
            )

        E, N, G, latent_dim = params.shape
        assert latent_dim == self.latent_dim, \
            f"Expected latent_dim={self.latent_dim}, got {latent_dim}"

        # Flatten to [E*N*G, latent_dim]
        z_flat = params.reshape(E * N * G, latent_dim)

        # Reshape for conv_transpose2d: [batch, latent_dim] → [batch, latent_dim, 1, 1]
        z_flat = z_flat.unsqueeze(-1).unsqueeze(-1)

        # Prepare labels if conditional
        if self.conditional:
            if labels is None:
                raise ValueError("Labels required for conditional GAN")

            # Expand labels to match seeds: [E, N] → [E, N, G] → [E*N*G]
            labels_expanded = labels.unsqueeze(2).expand(E, N, G).reshape(-1)

            # Generate with labels (no torch.no_grad() so gradients flow to latent codes)
            x_flat = self.generator(z_flat, labels_expanded)
        else:
            # Generate without labels (no torch.no_grad() so gradients flow to latent codes)
            x_flat = self.generator(z_flat)

        # Reshape back to [E, N, G, C, H, W]
        C, H, W = x_flat.shape[1:]
        reconstruction = x_flat.reshape(E, N, G, C, H, W)

        return self._apply_normalization(reconstruction)

    def prepare_for_stage(
        self,
        data_mean: torch.Tensor | None,
        data_std: torch.Tensor | None,
    ) -> None:
        """Set dataset normalization and move generator to the correct device.

        Device management happens here (in the setup phase) rather than
        lazily during forward() so that the forward pass remains a pure
        computation with no hidden side-effects.

        Args:
            data_mean: Per-channel mean [C, 1, 1]
            data_std:  Per-channel std  [C, 1, 1]
        """
        self.data_mean = data_mean
        self.data_std = data_std
        # Apply eval mode and freeze lazily here so ordering relative to
        # UnfrozenGANRepresentation.prepare_for_stage is well-defined.
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def _apply_normalization(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Convert GAN Tanh output [-1, 1] to dataset-normalized space.

        Pipeline: [-1, 1]  →  [0, 1]  →  (x - mean) / std

        If data_mean / data_std have not been set the tensor is returned
        unchanged (useful for sanity-checking the raw generator output).

        Args:
            reconstruction: [E, N, G, C, H, W] tensor with values in [-1, 1]

        Returns:
            Tensor in the same normalized space as the client's training data
        """
        if self.data_mean is None or self.data_std is None:
            return reconstruction

        # [-1, 1] → [0, 1]
        x = (reconstruction + 1.0) / 2.0

        # Broadcast mean/std [C, 1, 1] against reconstruction [*, C, H, W]
        mean = self.data_mean.to(x.device)
        std  = self.data_std.to(x.device)
        n_extra = x.ndim - mean.ndim
        for _ in range(n_extra):
            mean = mean.unsqueeze(0)
            std  = std.unsqueeze(0)

        return (x - mean) / std

    def get_parameter_shape(
        self,
        target_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Get latent parameter shape from target shape.

        Args:
            target_shape: Target shape [E, N, G, C, H, W]
            **kwargs: Ignored

        Returns:
            param_shape: [E, N, G, latent_dim]
        """
        # Extract E, N, G dimensions from target
        E, N, G = target_shape[:3]
        return (E, N, G, self.latent_dim)


class GIFDIntermediateRepresentation(RepresentationStrategy):
    """Representation for GIFD intermediate-feature optimisation stages.

    Takes intermediate features ``h_k`` at generator layer ``start_layer`` and
    runs the (frozen) generator from that layer to the output to produce
    pixel-space images.  This is used for stages 1 … K of GIFD (each stage
    optimises features at a progressively deeper layer).

    The input tensor is 6-D ``[E, N, G, C, H, W]`` (spatial features, not
    latent codes), which differs from :class:`GANRepresentation` whose input
    is ``[E, N, G, latent_dim]``.

    Args:
        generator: A :class:`~leakpro.fl_utils.gan_models.LayeredGANWrapper`
                   that supports :meth:`forward_from`.
        start_layer: Generator layer index from which to run (0-indexed,
                     inclusive).  Layer 0 is the initial ConvTranspose block.
        data_mean: Per-channel mean ``[C, 1, 1]`` for normalising GAN output.
        data_std:  Per-channel std  ``[C, 1, 1]`` for normalising GAN output.
    """

    def __init__(
        self,
        generator: nn.Module,
        start_layer: int,
        data_mean: torch.Tensor | None = None,
        data_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.start_layer = start_layer
        self.data_mean = data_mean
        self.data_std = data_std

        # Freeze generator – the *features* are the optimisable parameters.
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata."""
        return ComponentMetadata(
            name="gifd_intermediate_representation",
            required_capabilities={"has_pretrained_gan": True},
        )

    def forward(
        self,
        params: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode intermediate features to images via the partial generator.

        Args:
            params: Intermediate features ``[E, N, G, C, H, W]``.
            labels: Ignored (GIFD generator is unconditional).

        Returns:
            Reconstructed images ``[E, N, G, C_out, H_out, W_out]``.
        """
        if params.ndim != 6:
            raise ValueError(
                f"GIFDIntermediateRepresentation expects [E, N, G, C, H, W] (6-D), "
                f"but received shape {params.shape} ({params.ndim}-D). "
                f"Make sure the transition produces 6-D feature tensors."
            )

        E, N, G, C, H, W = params.shape

        # Flatten batch dims: [E, N, G, C, H, W] → [E*N*G, C, H, W]
        feat_flat = params.reshape(E * N * G, C, H, W)

        # Run partial generator (from start_layer to output)
        x_flat = self.generator.forward_from(feat_flat, start_layer=self.start_layer)

        # Reshape back: [E*N*G, C_out, H_out, W_out] → [E, N, G, C_out, H_out, W_out]
        C_out, H_out, W_out = x_flat.shape[1:]
        reconstruction = x_flat.reshape(E, N, G, C_out, H_out, W_out)

        return self._apply_normalization(reconstruction)

    def _apply_normalization(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Convert GAN Tanh output ``[-1, 1]`` to dataset-normalised space.

        Pipeline: ``[-1, 1] → [0, 1] → (x − mean) / std``

        If ``data_mean`` / ``data_std`` have not been set the tensor is
        returned unchanged (useful for sanity-checking raw generator output).
        """
        if self.data_mean is None or self.data_std is None:
            return reconstruction

        x = (reconstruction + 1.0) / 2.0
        mean = self.data_mean.to(x.device)
        std  = self.data_std.to(x.device)
        n_extra = x.ndim - mean.ndim
        for _ in range(n_extra):
            mean = mean.unsqueeze(0)
            std  = std.unsqueeze(0)
        return (x - mean) / std

    def prepare_for_stage(
        self,
        data_mean: torch.Tensor | None,
        data_std: torch.Tensor | None,
    ) -> None:
        """Store dataset normalisation parameters for :meth:`forward`."""
        self.data_mean = data_mean
        self.data_std = data_std

    def get_parameter_shape(self, target_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return *target_shape* unchanged.

        For GIFD intermediate stages the parameter shape equals the feature
        shape produced by the transition (already 6-D ``[E, N, G, C, H, W]``).
        """
        return target_shape


class UnfrozenGANRepresentation(GANRepresentation):
    """GAN representation that allows generator parameter optimization.

    Unlike GANRepresentation which freezes the generator, this variant
    allows fine-tuning the generator parameters during optimization.
    Used in GIAS stage 2 and CGIR.

    Args:
        generator: Pre-trained GAN generator model
        latent_dim: Dimension of latent code
        conditional: Whether generator is class-conditional
        num_classes: Number of classes (for conditional GANs)
    """

    def __init__(
        self,
        generator: nn.Module,
        latent_dim: int = 512,
        conditional: bool = False,
        num_classes: int | None = None,
    ) -> None:
        """Initialize unfrozen GAN representation.

        Args:
            generator: Pre-trained generator (will be made trainable)
            latent_dim: Dimension of latent code
            conditional: Whether generator accepts class labels
            num_classes: Number of classes (required if conditional=True)
        """
        # Don't call super().__init__ as it freezes the generator
        Component.__init__(self)
        self.generator = generator
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes
        # Must be initialised here (parent __init__ is skipped); prepare_for_stage() fills them.
        self.data_mean = None
        self.data_std = None

        # NOTE: do NOT call generator.train() here — the generator may be shared with
        # a preceding frozen stage (e.g. GIAS stage 1 GANRepresentation).  Switching
        # to train mode here would corrupt stage 1's eval-mode BatchNorm behaviour.
        # Instead, train mode is activated lazily in prepare_for_stage(), which is
        # called by GradientInversionBase.optimize() right before this stage runs.
        for param in self.generator.parameters():
            param.requires_grad = True

        if conditional and num_classes is None:
            raise ValueError("num_classes must be provided for conditional GANs")

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for this component."""
        return ComponentMetadata(
            name="unfrozen_gan_representation",
            required_capabilities={
                "has_pretrained_gan": True,
                "can_optimize_generator": True,
            },
        )

    def forward(
        self,
        params: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode latent codes to pixels via generator (with gradients).

        Args:
            params: Latent codes [E, N, G, latent_dim]
            labels: Class labels [E, N] (required if conditional=True)

        Returns:
            reconstruction: Generated images [E, N, G, C, H, W]
        """
        if params.ndim != 4:
            raise ValueError(
                f"UnfrozenGANRepresentation expects latent codes with shape [E, N, G, latent_dim] (4D), "
                f"but got tensor with shape {params.shape} ({params.ndim}D). "
                f"This usually means pixel-space data was passed instead of latent codes."
            )

        E, N, G, latent_dim = params.shape
        assert latent_dim == self.latent_dim, \
            f"Expected latent_dim={self.latent_dim}, got {latent_dim}"

        # Flatten to [E*N*G, latent_dim]
        z_flat = params.reshape(E * N * G, latent_dim)
        
        # Reshape for conv_transpose2d: [batch, latent_dim] → [batch, latent_dim, 1, 1]
        z_flat = z_flat.unsqueeze(-1).unsqueeze(-1)

        # Prepare labels if conditional
        if self.conditional:
            if labels is None:
                raise ValueError("Labels required for conditional GAN")

            labels_expanded = labels.unsqueeze(2).expand(E, N, G).reshape(-1)
            # Generate WITH gradients (no torch.no_grad())
            x_flat = self.generator(z_flat, labels_expanded)
        else:
            # Generate WITH gradients
            x_flat = self.generator(z_flat)

        # Reshape back to [E, N, G, C, H, W]
        C, H, W = x_flat.shape[1:]
        reconstruction = x_flat.reshape(E, N, G, C, H, W)

        return self._apply_normalization(reconstruction)


    def prepare_for_stage(
        self,
        data_mean: torch.Tensor | None,
        data_std: torch.Tensor | None,
    ) -> None:
        """Set normalization and unfreeze generator parameters for fine-tuning.

        The generator remains in **eval mode** (BatchNorm uses running statistics)
        while all parameters are made trainable.  This matches the original GIAS
        implementation which calls ``G.requires_grad_(True)`` but never calls
        ``G.train()``, so BatchNorm normalization stays stable across stage 2.

        Called by GradientInversionBase.optimize() at the start of this stage,
        after the preceding frozen stage has already completed.
        """
        self.data_mean = data_mean
        self.data_std = data_std
        # Keep eval mode so BatchNorm uses stable running stats (not batch stats).
        # Gradients still flow to affine parameters (weight/bias) of BatchNorm.
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = True


