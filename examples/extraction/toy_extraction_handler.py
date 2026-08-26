"""Runnable LeakPro provider used to smoke-test both extraction attacks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from leakpro import AbstractExtractionInputHandler
from leakpro.attacks.extraction_attacks.adapters import CallableDiffusionAdapter
from leakpro.attacks.extraction_attacks.protocols import ConditionGradient


class TwoFeatureExtractor(nn.Module):
    """Map dark and bright images to two non-zero feature directions."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return two brightness-derived features."""
        mean = images.mean(dim=(1, 2, 3))
        return torch.stack((mean, 1.0 - mean), dim=1)


class TinyTimeClassifier(nn.Module):
    """Keep the runnable integration smoke test fast."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.image = nn.Linear(in_channels * 4 * 4, num_classes)
        self.time = nn.Linear(1, num_classes, bias=False)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return cluster logits from image pixels and time."""
        return self.image(images.flatten(start_dim=1)) + self.time(timesteps.float().unsqueeze(1) / 10.0)


class ToyExtractionHandler(AbstractExtractionInputHandler):
    """Supply a tiny authorized target through LeakPro's extraction boundary."""

    def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
        """Return a deterministic adapter with forward and guided operations."""

        def sample(batch_size: int, conditions: Sequence[object] | None, seed: int) -> torch.Tensor:
            del seed
            if conditions is not None:
                images = torch.zeros((batch_size, 1, 4, 4))
                images[-2] = 0.5
                images[-1] = 1.0
                return images
            values = torch.arange(batch_size).remainder(2).float()
            return values.view(-1, 1, 1, 1).expand(-1, 1, 4, 4).clone()

        def q_sample(clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            scale = timesteps.float().view(-1, 1, 1, 1) / 100.0
            return clean + scale * noise

        def guided_sample(
            batch_size: int,
            labels: torch.Tensor,
            gradient_fn: ConditionGradient,
            seed: int,
        ) -> torch.Tensor:
            del seed
            noisy = torch.full((batch_size, 1, 4, 4), 0.5)
            gradient = gradient_fn(noisy, torch.ones(batch_size, dtype=torch.long), labels)
            if gradient.shape != noisy.shape or not torch.isfinite(gradient).all():
                raise RuntimeError("Invalid guidance gradient in toy adapter.")
            values = torch.arange(batch_size).remainder(2).float()
            return values.view(-1, 1, 1, 1).expand(-1, 1, 4, 4).clone()

        return CallableDiffusionAdapter(
            image_shape=(1, 4, 4),
            sample_fn=sample,
            num_timesteps=10,
            q_sample_fn=q_sample,
            guided_sample_fn=guided_sample,
        )

    def get_extraction_conditions(self) -> Sequence[object]:
        """Return one repeated conditional query."""
        return ["memorized"]

    def get_extraction_reference_images(self) -> torch.Tensor:
        """Return two authorized toy references."""
        return torch.stack((torch.zeros((1, 4, 4)), torch.ones((1, 4, 4))))

    def get_side_feature_extractor(self) -> nn.Module:
        """Return the toy frozen feature map."""
        return TwoFeatureExtractor()

    def get_side_classifier_factory(self) -> Callable[[int, int], nn.Module]:
        """Return a tiny classifier factory for the smoke test."""
        return TinyTimeClassifier
