#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Simple GAN model architectures for gradient inversion attacks.

This module provides basic GAN architectures that can be trained from scratch
or used as baselines.
"""

from __future__ import annotations

import torch
from torch import nn


class DCGANGenerator(nn.Module):
    """DCGAN generator for image synthesis.

    Args:
        latent_dim: Dimension of latent code
        img_channels: Number of output image channels
        feature_maps: Base number of feature maps
        img_size: Output image size (must be power of 2)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        img_channels: int = 3,
        feature_maps: int = 64,
        img_size: int = 64,
    ) -> None:
        """Initialize DCGAN generator."""
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        # Calculate number of upsampling layers
        num_layers = 0
        size = 4
        while size < img_size:
            size *= 2
            num_layers += 1

        # Build network
        layers = []

        # Initial projection: latent_dim -> feature_maps*8 * 4 * 4
        current_features = feature_maps * (2 ** (num_layers + 1))
        layers.extend([
            nn.ConvTranspose2d(latent_dim, current_features, 4, 1, 0, bias=False),
            nn.BatchNorm2d(current_features),
            nn.ReLU(True),
        ])

        # Upsampling layers
        for i in range(num_layers):
            next_features = current_features // 2
            layers.extend([
                nn.ConvTranspose2d(current_features, next_features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_features),
                nn.ReLU(True),
            ])
            current_features = next_features

        # Final layer
        layers.extend([
            nn.ConvTranspose2d(current_features, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        ])

        self.main = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Generate images from latent codes.

        Args:
            z: Latent codes [batch_size, latent_dim]
            labels: Optional labels (unused in unconditional model)

        Returns:
            Generated images [batch_size, img_channels, img_size, img_size]
        """
        # Reshape to [batch, latent_dim, 1, 1]
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)


class ConditionalDCGANGenerator(nn.Module):
    """Conditional DCGAN generator with class labels.

    Args:
        latent_dim: Dimension of latent code
        num_classes: Number of classes
        img_channels: Number of output image channels
        feature_maps: Base number of feature maps
        img_size: Output image size (must be power of 2)
        embed_dim: Dimension of label embedding
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        img_channels: int = 3,
        feature_maps: int = 64,
        img_size: int = 32,
        embed_dim: int = 50,
    ) -> None:
        """Initialize conditional DCGAN generator."""
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Calculate number of upsampling layers
        num_layers = 0
        size = 4
        while size < img_size:
            size *= 2
            num_layers += 1

        # Build network (takes concatenated [z, embed] as input)
        input_dim = latent_dim + embed_dim
        layers = []

        # Initial projection
        current_features = feature_maps * (2 ** (num_layers + 1))
        layers.extend([
            nn.ConvTranspose2d(input_dim, current_features, 4, 1, 0, bias=False),
            nn.BatchNorm2d(current_features),
            nn.ReLU(True),
        ])

        # Upsampling layers
        for i in range(num_layers):
            next_features = current_features // 2
            layers.extend([
                nn.ConvTranspose2d(current_features, next_features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_features),
                nn.ReLU(True),
            ])
            current_features = next_features

        # Final layer
        layers.extend([
            nn.ConvTranspose2d(current_features, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        ])

        self.main = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate images from latent codes and labels.

        Args:
            z: Latent codes [batch_size, latent_dim]
            labels: Class labels [batch_size]

        Returns:
            Generated images [batch_size, img_channels, img_size, img_size]
        """
        # Embed labels
        label_embed = self.label_embedding(labels)  # [batch, embed_dim]

        # Concatenate z and label embedding
        z_combined = torch.cat([z, label_embed], dim=1)  # [batch, latent_dim + embed_dim]

        # Reshape to [batch, input_dim, 1, 1]
        z_combined = z_combined.view(z_combined.size(0), z_combined.size(1), 1, 1)

        return self.main(z_combined)


class DCGANDiscriminator(nn.Module):
    """DCGAN discriminator for image classification.

    Args:
        img_channels: Number of input image channels
        feature_maps: Base number of feature maps
        img_size: Input image size (must be power of 2)
    """

    def __init__(
        self,
        img_channels: int = 3,
        feature_maps: int = 64,
        img_size: int = 64,
    ) -> None:
        """Initialize DCGAN discriminator."""
        super().__init__()
        self.img_size = img_size

        # For CIFAR-10 (32x32), we need exactly 3 stride-2 convs to reach 4x4:
        # 32 -> 16 -> 8 -> 4, then final conv 4 -> 1
        
        layers = []
        
        if img_size == 32:
            # Architecture for 32x32 images (CIFAR-10)
            layers.extend([
                # 32 -> 16
                nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 16 -> 8
                nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 8 -> 4
                nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 4 -> 1
                nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            ])
        elif img_size == 64:
            # Architecture for 64x64 images
            layers.extend([
                # 64 -> 32
                nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 32 -> 16
                nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 16 -> 8
                nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 8 -> 4
                nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # 4 -> 1
                nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            ])
        else:
            raise ValueError(f"Unsupported image size: {img_size}. Use 32 or 64.")
        
        self.main = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Classify images as real or fake.

        Args:
            img: Input images [batch_size, img_channels, img_size, img_size]

        Returns:
            Predictions [batch_size] (probability of being real)
        """
        out = self.main(img)  # [batch_size, 1, 1, 1]
        out = out.view(-1)  # [batch_size]
        return self.sigmoid(out)


class LayeredGANWrapper(nn.Module):
    """Generator wrapper enabling layer-by-layer execution for GIFD.

    Splits a generator's ``main`` Sequential into ConvTranspose2d-led blocks
    (one block per semantic "level"), and exposes:

    * :meth:`forward_from` – run from a specific layer index to the output
    * :meth:`apply_layer` – run exactly one layer block (used by transitions)
    * :meth:`forward` – standard end-to-end forward pass (drop-in replacement)

    The wrapper is compatible with :class:`GANRepresentation` for latent-space
    stage 0 and with :class:`GIFDIntermediateRepresentation` for intermediate
    feature stages 1 … K.

    Args:
        generator: Base generator with a ``main: nn.Sequential`` attribute
                   (e.g. :class:`DCGANGenerator`).

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
          attribute (e.g. :class:`DCGANGenerator`).
        * ``generator`` itself is an ``nn.Sequential`` – the generator returned
          by :func:`~leakpro.fl_utils.gan_handler._build_dcgan_generator` is a
          bare ``nn.Sequential`` (used by the HuggingFace / GitHub loader).
        """
        if isinstance(self.generator, nn.Sequential):
            sequential = self.generator
        elif hasattr(self.generator, "main") and isinstance(self.generator.main, nn.Sequential):
            sequential = self.generator.main
        else:
            raise ValueError(
                "LayeredGANWrapper requires either a bare nn.Sequential generator "
                "or a generator with a 'main: nn.Sequential' attribute "
                "(DCGANGenerator and the csinva HuggingFace DCGAN are both compatible)."
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

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
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
        """Apply exactly one layer group (used by :class:`GIFDLayerTransition`).

        Args:
            features: Input features ``[batch, C, H, W]``
                      (or ``[batch, latent_dim, 1, 1]`` for layer 0).
            layer_idx: Index of the layer group to apply.

        Returns:
            Output features ``[batch, C', H', W']``.
        """
        return self.layer_groups[layer_idx](features)

    def get_feature_shape(
        self,
        layer_idx: int,
        latent_dim: int | None = None,
        device: torch.device | None = None,
    ) -> tuple[int, ...]:
        """Return the spatial feature shape produced by layer *layer_idx*.

        Performs a single dry-run forward pass with a random latent code.
        The result is ``(C, H, W)`` – i.e., the shape of the intermediate
        features **after** applying layers 0 … layer_idx.

        Args:
            layer_idx: Target layer index (0-indexed).
            latent_dim: Latent dimension (defaults to ``self.latent_dim``).
            device: Device for the dry-run tensor.

        Returns:
            ``(C, H, W)`` shape tuple.
        """
        latent_dim = latent_dim or self.latent_dim or 100
        dev = device or next(self.generator.parameters()).device
        with torch.no_grad():
            z = torch.zeros(1, latent_dim, 1, 1, device=dev)
            x = z
            for i in range(layer_idx + 1):
                x = self.layer_groups[i](x)
        return tuple(x.shape[1:])  # (C, H, W)


class BigGANLayeredWrapper(nn.Module):
    """Wraps ``pytorch_pretrained_biggan.BigGAN`` with a LayeredGANWrapper-compatible API.

    Enables both simple forward passes (for GIAS) and layer-by-layer execution
    (for GIFD).  The wrapper is conditional: it accepts class-label indices and
    converts them to one-hot vectors internally.

    Args:
        biggan_model: A ``BigGAN`` instance from ``pytorch_pretrained_biggan``.
        truncation:   Truncation factor ∈ (0, 1] (default 0.4).
        img_size:     Spatial size to resize output to (default 64).
    """

    def __init__(
        self,
        biggan_model: nn.Module,
        truncation: float = 0.4,
        img_size: int = 64,
    ) -> None:
        super().__init__()
        self.biggan = biggan_model
        self.truncation = truncation
        self.img_size = img_size
        self.latent_dim: int = biggan_model.config.z_dim
        # num_layer_groups counts all entries in Generator.layers
        # (GenBlocks + SelfAttn), matching the GIFD paper's stage count.
        self.num_layer_groups: int = len(biggan_model.generator.layers)

        # Mutable state set on forward() / set_labels(); reused by all GIFD stages
        self._stored_cond: torch.Tensor | None = None
        self._stored_labels: torch.Tensor | None = None  # one-hot [B, num_classes]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_labels(self, labels: torch.Tensor) -> None:
        """Store class labels (long [B]) for conditional generation."""
        num_classes = self.biggan.config.num_classes
        dev = next(self.biggan.parameters()).device
        self._stored_labels = torch.nn.functional.one_hot(
            labels.long().to(dev), num_classes
        ).float()

    def _compute_cond(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        """Compute condition vector cat(z, embed) [B, z_dim*2]."""
        embed = self.biggan.embeddings(y_onehot)
        return torch.cat((z, embed), dim=1)

    def _gen_init_features(self, cond_vector: torch.Tensor) -> torch.Tensor:
        """Project cond_vector through gen_z and reshape to spatial format."""
        gen = self.biggan.generator
        ch = self.biggan.config.channel_width
        h = gen.gen_z(cond_vector)           # [B, 16*ch*4*4]
        h = h.view(-1, 4, 4, 16 * ch)
        h = h.permute(0, 3, 1, 2).contiguous()  # [B, 16*ch, 4, 4]
        return h

    def _run_layers(
        self,
        h: torch.Tensor,
        cond_vector: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Run generator layers[start:end] with duck-typed dispatch."""
        for layer in self.biggan.generator.layers[start:end]:
            try:
                h = layer(h, cond_vector, self.truncation)
            except TypeError:
                # SelfAttn and other single-argument layers
                h = layer(h)
        return h

    def _finalize(self, h: torch.Tensor) -> torch.Tensor:
        """Apply final BN → ReLU → conv_to_rgb → tanh → resize.

        Uses ``mode='area'`` for downsampling, matching the GIFD paper's
        ``gen_dummy_data()`` which calls
        ``F.interpolate(..., mode='area')``.
        """
        gen = self.biggan.generator
        h = gen.bn(h, self.truncation)
        h = gen.relu(h)
        h = gen.conv_to_rgb(h)
        h = h[:, :3, ...]
        h = gen.tanh(h)
        if h.shape[-1] != self.img_size:
            h = torch.nn.functional.interpolate(
                h, size=(self.img_size, self.img_size), mode="area"
            )
        return h

    def _expand_to_batch(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Repeat tensor along dim-0 to match batch_size."""
        if t.shape[0] != batch_size:
            repeats = (batch_size + t.shape[0] - 1) // t.shape[0]
            t = t.repeat(repeats, *([1] * (t.ndim - 1)))[:batch_size]
        return t

    # ------------------------------------------------------------------
    # Public LayeredGANWrapper-compatible API
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Full latent → image forward (used in GIAS / GIFD stage 0).

        Args:
            z:      Latent codes [B, latent_dim] or [B, latent_dim, 1, 1].
            labels: Class indices [B] (long).  Required on first call;
                    reuses stored labels if omitted.

        Returns:
            Generated images [B, 3, img_size, img_size].
        """
        if z.ndim == 4:
            z = z.squeeze(-1).squeeze(-1)
        if labels is not None:
            self.set_labels(labels)
        if self._stored_labels is None:
            raise RuntimeError(
                "BigGANLayeredWrapper: labels required.  Call set_labels() or pass labels= to forward()."
            )
        B = z.shape[0]
        y = self._expand_to_batch(self._stored_labels, B)
        cond = self._compute_cond(z, y)
        self._stored_cond = cond.detach()
        h = self._gen_init_features(cond)
        h = self._run_layers(h, cond, 0, self.num_layer_groups)
        return self._finalize(h)

    def forward_from(self, features: torch.Tensor, start_layer: int) -> torch.Tensor:
        """Run the generator tail from *start_layer* onward (GIFD intermediate stages).

        Requires a previous :meth:`forward` call to have stored ``_stored_cond``.

        Args:
            features:    Intermediate feature map [B, C, H, W].
            start_layer: First layer to apply (0 = run full pass).

        Returns:
            Pixel-space images [B, 3, img_size, img_size].
        """
        if start_layer == 0:
            return self.forward(features)
        if self._stored_cond is None:
            raise RuntimeError(
                "BigGANLayeredWrapper.forward_from: no stored cond_vector.  Run forward() first."
            )
        B = features.shape[0]
        cond = self._expand_to_batch(self._stored_cond, B)
        h = self._run_layers(features, cond, start_layer, self.num_layer_groups)
        return self._finalize(h)

    def apply_layer(self, features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply a single generator layer (used by GIFDLayerTransition).

        For ``layer_idx == 0``: takes latent code z, runs gen_z + layers[0].
        For ``layer_idx > 0``: takes intermediate features, runs layers[layer_idx].

        Args:
            features: Latent codes or feature map.
            layer_idx: Layer index (0-indexed into generator.layers).

        Returns:
            Feature map after applying the layer.
        """
        gen = self.biggan.generator
        if layer_idx == 0:
            if features.ndim == 4:
                features = features.squeeze(-1).squeeze(-1)
            if self._stored_labels is None:
                raise RuntimeError(
                    "BigGANLayeredWrapper.apply_layer: labels required.  Call set_labels() first."
                )
            B = features.shape[0]
            y = self._expand_to_batch(self._stored_labels, B)
            cond = self._compute_cond(features, y)
            self._stored_cond = cond.detach()
            h = self._gen_init_features(cond)
            layer = gen.layers[0]
            try:
                h = layer(h, cond, self.truncation)
            except TypeError:
                h = layer(h)
        else:
            if self._stored_cond is None:
                raise RuntimeError(
                    "BigGANLayeredWrapper.apply_layer: no stored cond_vector.  Run layer 0 first."
                )
            B = features.shape[0]
            cond = self._expand_to_batch(self._stored_cond, B)
            layer = gen.layers[layer_idx]
            try:
                h = layer(features, cond, self.truncation)
            except TypeError:
                h = layer(features)
        return h

    def get_feature_shape(
        self,
        layer_idx: int,
        latent_dim: int | None = None,
        device: torch.device | None = None,
    ) -> tuple[int, ...]:
        """Dry-run forward to return feature shape after *layer_idx*."""
        latent_dim = latent_dim or self.latent_dim
        dev = device or next(self.biggan.parameters()).device
        saved_cond, saved_labels = self._stored_cond, self._stored_labels
        try:
            with torch.no_grad():
                z = torch.zeros(1, latent_dim, device=dev)
                y = torch.zeros(1, self.biggan.config.num_classes, device=dev)
                y[0, 0] = 1.0
                self._stored_labels = y
                h = z
                for i in range(layer_idx + 1):
                    h = self.apply_layer(h, i)
            return tuple(h.shape[1:])
        finally:
            self._stored_cond, self._stored_labels = saved_cond, saved_labels


__all__ = [
    "DCGANGenerator",
    "ConditionalDCGANGenerator",
    "DCGANDiscriminator",
    "LayeredGANWrapper",
    "BigGANLayeredWrapper",
]
