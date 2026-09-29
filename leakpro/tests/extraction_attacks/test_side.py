"""SIDE classifier, clustering, guidance, and reference-metric tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch import nn

from leakpro.attacks.extraction_attacks.adapters import CallableDiffusionAdapter
from leakpro.attacks.extraction_attacks.side import AttackSIDEExtraction


class TwoFeatureExtractor(nn.Module):
    """Map dark and bright toy images to orthogonal non-zero features."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.mean(dim=(1, 2, 3))
        return torch.stack([mean, 1.0 - mean], dim=1)


class TinyTimeClassifier(nn.Module):
    """Small differentiable time-conditioned classifier for smoke testing."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.image = nn.Linear(in_channels * 4 * 4, num_classes)
        self.time = nn.Linear(1, num_classes, bias=False)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time = timesteps.to(dtype=images.dtype).unsqueeze(1) / 10.0
        return self.image(images.flatten(start_dim=1)) + self.time(time)


def test_side_default_uses_raw_feature_space_kmeans() -> None:
    adapter = CallableDiffusionAdapter(
        image_shape=(1, 1, 1),
        sample_fn=lambda batch_size, conditions, seed: torch.zeros((batch_size, 1, 1, 1)),
    )
    attack = AttackSIDEExtraction(
        adapter,
        TwoFeatureExtractor(),
        {
            "authorized_audit": True,
            "synthetic_samples": 6,
            "clusters": 2,
            "min_cluster_size": 1,
            "cohesion_threshold": -1.0,
        },
        audit_hash="raw-kmeans-test",
    )
    attack._synthetic_features = torch.tensor(  # noqa: SLF001 - direct algorithm boundary test
        [[1.0, 0.0], [2.0, 0.0], [100.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]
    )

    attack._fit_surrogate_clusters()  # noqa: SLF001 - direct algorithm boundary test

    assert attack.synthetic_labels is not None
    assert attack.synthetic_labels[0] != attack.synthetic_labels[2]
    assert attack.synthetic_labels[0] == attack.synthetic_labels[3]


def test_side_reassigns_samples_from_a_rejected_cluster() -> None:
    adapter = CallableDiffusionAdapter(
        image_shape=(1, 1, 1),
        sample_fn=lambda batch_size, conditions, seed: torch.zeros((batch_size, 1, 1, 1)),
    )
    attack = AttackSIDEExtraction(
        adapter,
        TwoFeatureExtractor(),
        {
            "authorized_audit": True,
            "synthetic_samples": 6,
            "clusters": 3,
            "min_cluster_size": 1,
            "cohesion_threshold": 0.5,
        },
        audit_hash="reassignment-test",
    )
    attack._synthetic_features = torch.tensor(  # noqa: SLF001 - fixed clustering input
        [[100.0, 0.0], [100.0, 0.0], [0.0, 100.0], [0.0, 100.0], [1.0, 0.0], [-1.0, 0.0]]
    )

    attack._fit_surrogate_clusters()  # noqa: SLF001 - direct algorithm boundary test

    assert len(attack.cluster_cohesion) == 2
    assert attack.synthetic_labels is not None
    assert attack.synthetic_labels.shape == (6,)
    assert attack.synthetic_labels[4] == attack.synthetic_labels[0]
    assert attack.synthetic_labels[5] == attack.synthetic_labels[2]


def test_side_l2_results_use_zero_one_coordinates_for_minus_one_one_model() -> None:
    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        del conditions, seed
        values = torch.arange(batch_size).remainder(2).float().mul(2).sub(1)
        return values.view(-1, 1, 1, 1).expand(-1, 1, 4, 4).clone()

    def q_sample(clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        del timesteps, noise
        return clean

    def guided_sample(
        batch_size: int,
        labels: torch.Tensor,
        gradient_fn: Any,
        seed: int,
    ) -> torch.Tensor:
        del seed
        noisy = torch.full((batch_size, 1, 4, 4), 0.38)
        gradient_fn(noisy, torch.ones(batch_size, dtype=torch.long), labels)
        return torch.full((batch_size, 1, 4, 4), 0.38)

    adapter = CallableDiffusionAdapter(
        image_shape=(1, 4, 4),
        sample_fn=sample,
        num_timesteps=10,
        q_sample_fn=q_sample,
        guided_sample_fn=guided_sample,
    )
    attack = AttackSIDEExtraction(
        adapter,
        TwoFeatureExtractor(),
        {
            "authorized_audit": True,
            "image_range": "minus_one_one",
            "compute_device": "cpu",
            "synthetic_samples": 8,
            "synthetic_batch_size": 8,
            "clusters": 2,
            "cohesion_threshold": 0.9,
            "min_cluster_size": 2,
            "classifier_epochs": 1,
            "classifier_batch_size": 4,
            "num_generations": 1,
            "generation_batch_size": 1,
            "l2_bands": {"near": {"lower": 0.675, "upper": 0.7}},
        },
        audit_hash="side-minus-one-one-range",
        reference_images=torch.full((1, 1, 4, 4), -1.0),
        classifier_factory=TinyTimeClassifier,
    )

    attack.prepare_attack()
    result = attack.run_attack()

    assert result.metrics["nearest_l2_mean"] == pytest.approx(0.69)
    assert result.metrics["l2_coordinate_range"] == "zero_one"
    assert result.metrics["l2_band_scores"]["near"]["ams"] == 1.0
    assert result.metrics["l2_band_scores"]["near"]["ums"] == 1.0
    assert result.candidates[0].nearest_reference_distance == pytest.approx(0.69)
    assert result.candidates[0].score == pytest.approx(0.69)
    torch.testing.assert_close(result.images, torch.full_like(result.images, 0.69))
