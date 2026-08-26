"""SIDE classifier, clustering, guidance, and reference-metric tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch import nn

import leakpro.attacks.extraction_attacks.side as side_module
from leakpro.attacks.extraction_attacks.abstract_extraction import AttackState
from leakpro.attacks.extraction_attacks.adapters import CallableDiffusionAdapter
from leakpro.attacks.extraction_attacks.classifier import TimeConditionedResNet
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


class DtypeRecordingFeatureExtractor(nn.Module):
    """Record the floating dtype presented to a parameterized extractor."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=dtype))
        self.observed_dtypes: list[torch.dtype] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.observed_dtypes.append(images.dtype)
        mean = images.mean(dim=(1, 2, 3)) * self.scale
        return torch.stack([mean, 1.0 - mean], dim=1)


class MixedDtypeFeatureExtractor(nn.Module):
    """Expose an ambiguous mixed-floating-dtype extractor state."""

    def __init__(self) -> None:
        super().__init__()
        self.half_scale = nn.Parameter(torch.ones((), dtype=torch.float16))
        self.double_scale = nn.Parameter(torch.ones((), dtype=torch.float64))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3)).unsqueeze(1)


def _small_side_adapter(
    *,
    image_range: str = "zero_one",
    sample_calls: list[int] | None = None,
) -> CallableDiffusionAdapter:
    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        del conditions, seed
        if sample_calls is not None:
            sample_calls.append(batch_size)
        values = torch.arange(batch_size).remainder(2).float()
        if image_range == "minus_one_one":
            values = values.mul(2).sub(1)
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
        center = 0.0 if image_range == "minus_one_one" else 0.5
        noisy = torch.full((batch_size, 1, 4, 4), center)
        gradient_fn(noisy, torch.ones(batch_size, dtype=torch.long), labels)
        return sample(batch_size, None, 0)

    return CallableDiffusionAdapter(
        image_shape=(1, 4, 4),
        sample_fn=sample,
        num_timesteps=10,
        q_sample_fn=q_sample,
        guided_sample_fn=guided_sample,
    )


def _small_side_config(*, image_range: str = "zero_one") -> dict[str, Any]:
    return {
        "authorized_audit": True,
        "image_range": image_range,
        "compute_device": "cpu",
        "synthetic_samples": 4,
        "synthetic_batch_size": 4,
        "clusters": 2,
        "cohesion_threshold": -1.0,
        "min_cluster_size": 1,
        "classifier_epochs": 1,
        "classifier_batch_size": 2,
        "num_generations": 4,
        "generation_batch_size": 2,
    }


def test_time_conditioned_resnet_shape_and_input_gradient() -> None:
    classifier = TimeConditionedResNet(
        in_channels=1,
        num_classes=3,
        base_width=4,
        blocks=(1, 1, 1, 1),
        timestep_embedding_dim=8,
    )
    images = torch.randn(2, 1, 8, 8, requires_grad=True)
    logits = classifier(images, torch.tensor([0, 5]))
    assert logits.shape == (2, 3)
    logits.sum().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()


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
        audit_fingerprint="raw-kmeans-test",
    )
    attack._synthetic_features = torch.tensor(  # noqa: SLF001 - direct algorithm boundary test
        [[1.0, 0.0], [2.0, 0.0], [100.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]
    )

    attack._fit_surrogate_clusters()  # noqa: SLF001 - direct algorithm boundary test

    assert attack.synthetic_labels is not None
    assert attack.synthetic_labels[0] != attack.synthetic_labels[2]
    assert attack.synthetic_labels[0] == attack.synthetic_labels[3]


def test_side_end_to_end_with_toy_diffusion_adapter() -> None:
    references = torch.stack([torch.zeros((1, 4, 4)), torch.ones((1, 4, 4))])

    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        del conditions, seed
        values = torch.arange(batch_size).remainder(2).float()
        return values.view(-1, 1, 1, 1).expand(-1, 1, 4, 4).clone()

    def q_sample(clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        scale = timesteps.float().view(-1, 1, 1, 1) / 100.0
        return clean + scale * noise

    def guided_sample(
        batch_size: int,
        labels: torch.Tensor,
        gradient_fn: Any,
        seed: int,
    ) -> torch.Tensor:
        del seed
        noisy = torch.full((batch_size, 1, 4, 4), 0.5)
        timesteps = torch.ones(batch_size, dtype=torch.long)
        gradient = gradient_fn(noisy, timesteps, labels)
        assert gradient.shape == noisy.shape and torch.isfinite(gradient).all()
        values = torch.arange(batch_size).remainder(2).float()
        return values.view(-1, 1, 1, 1).expand(-1, 1, 4, 4).clone()

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
            "compute_device": "cpu",
            "synthetic_samples": 8,
            "synthetic_batch_size": 8,
            "clusters": 2,
            "cohesion_threshold": 0.9,
            "min_cluster_size": 2,
            "classifier_epochs": 1,
            "classifier_batch_size": 4,
            "classifier_learning_rate": 1e-3,
            "num_generations": 4,
            "generation_batch_size": 4,
            "guidance_scale": 1.0,
            "l2_bands": {"high": {"lower": 0.0, "upper": 0.01}},
        },
        audit_fingerprint="side-test",
        reference_images=references,
        classifier_factory=TinyTimeClassifier,
    )
    torch.manual_seed(913)
    rng_state = torch.random.get_rng_state()
    attack.prepare_attack()
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    result = attack.run_attack()
    assert result.images.shape == (4, 1, 4, 4)
    assert result.metrics["retained_clusters"] == 2
    assert result.metrics["l2_band_scores"]["high"]["ams"] == 1.0
    assert result.metrics["l2_band_scores"]["high"]["ums"] == 0.5
    assert len(result.metrics["classifier_epoch_losses"]) == 1


def test_side_converts_features_to_extractor_dtype_before_inference() -> None:
    extractor = DtypeRecordingFeatureExtractor(torch.float16)
    attack = AttackSIDEExtraction(
        _small_side_adapter(),
        extractor,
        _small_side_config(),
        audit_fingerprint="feature-extractor-dtype-test",
        classifier_factory=TinyTimeClassifier,
    )

    attack.prepare_attack()

    assert extractor.observed_dtypes == [torch.float16]


def test_side_rejects_mixed_feature_extractor_dtypes_before_sampling() -> None:
    sample_calls: list[int] = []
    attack = AttackSIDEExtraction(
        _small_side_adapter(sample_calls=sample_calls),
        MixedDtypeFeatureExtractor(),
        _small_side_config(),
        audit_fingerprint="mixed-feature-extractor-dtype-test",
        classifier_factory=TinyTimeClassifier,
    )

    with pytest.raises(ValueError, match="feature-extractor floating-point"):
        attack.prepare_attack()

    assert sample_calls == []


def test_failed_side_preparation_cannot_reuse_partial_sampling_state() -> None:
    sample_calls: list[int] = []

    class NonfiniteFeatureExtractor(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.full((images.shape[0], 2), torch.nan, device=images.device)

    attack = AttackSIDEExtraction(
        _small_side_adapter(sample_calls=sample_calls),
        NonfiniteFeatureExtractor(),
        _small_side_config(),
        audit_fingerprint="failed-preparation-lifecycle-test",
        classifier_factory=TinyTimeClassifier,
    )

    with pytest.raises(ValueError, match="NaN or infinity"):
        attack.prepare_attack()
    assert attack.state is AttackState.FAILED
    assert sample_calls == [4]

    with pytest.raises(RuntimeError, match="one-shot"):
        attack.prepare_attack()
    assert sample_calls == [4]


def test_reference_free_side_does_not_concatenate_raw_metric_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attack = AttackSIDEExtraction(
        _small_side_adapter(image_range="minus_one_one"),
        TwoFeatureExtractor(),
        _small_side_config(image_range="minus_one_one"),
        audit_fingerprint="reference-free-allocation-test",
        classifier_factory=TinyTimeClassifier,
    )
    attack.prepare_attack()
    real_cat = torch.cat
    image_concatenations = 0

    def tracked_cat(tensors: Sequence[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal image_concatenations
        tensors = tuple(tensors)
        if tensors and tensors[0].ndim == 4:
            image_concatenations += 1
        return real_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(side_module.torch, "cat", tracked_cat)

    attack.run_attack()

    assert image_concatenations == 1


def test_zero_one_reference_metrics_reuse_persisted_image_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attack = AttackSIDEExtraction(
        _small_side_adapter(),
        TwoFeatureExtractor(),
        _small_side_config(),
        audit_fingerprint="zero-one-reference-allocation-test",
        reference_images=torch.stack((torch.zeros((1, 4, 4)), torch.ones((1, 4, 4)))),
        classifier_factory=TinyTimeClassifier,
    )
    attack.prepare_attack()
    real_nearest_reference = side_module.nearest_reference
    metric_inputs: list[torch.Tensor] = []

    def tracked_nearest_reference(
        candidates: torch.Tensor,
        references: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metric_inputs.append(candidates)
        return real_nearest_reference(candidates, references, **kwargs)

    monkeypatch.setattr(side_module, "nearest_reference", tracked_nearest_reference)

    result = attack.run_attack()

    assert len(metric_inputs) == 1
    assert metric_inputs[0].data_ptr() == result.images.data_ptr()


@pytest.mark.parametrize("missing", ["q_sample_fn", "guided_sample_fn"])
def test_side_rejects_missing_white_box_capability_before_sampling(missing: str) -> None:
    sample_calls = 0

    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        nonlocal sample_calls
        del conditions, seed
        sample_calls += 1
        return torch.zeros((batch_size, 1, 4, 4))

    q_sample = lambda clean, timesteps, noise: clean + noise * 0  # noqa: E731
    guided_sample = lambda batch_size, labels, gradient_fn, seed: torch.zeros((batch_size, 1, 4, 4))  # noqa: E731
    adapter = CallableDiffusionAdapter(
        image_shape=(1, 4, 4),
        sample_fn=sample,
        num_timesteps=10,
        q_sample_fn=None if missing == "q_sample_fn" else q_sample,
        guided_sample_fn=None if missing == "guided_sample_fn" else guided_sample,
    )
    attack = AttackSIDEExtraction(
        adapter,
        TwoFeatureExtractor(),
        {
            "authorized_audit": True,
            "synthetic_samples": 4,
            "clusters": 2,
            "min_cluster_size": 1,
            "cohesion_threshold": -1.0,
        },
        audit_fingerprint=f"missing-{missing}",
    )

    with pytest.raises(ValueError, match=missing):
        attack.prepare_attack()

    assert sample_calls == 0


@pytest.mark.parametrize(
    "malformed",
    ["sample_fn", "q_sample_fn", "guided_sample_fn", "classifier_timestep_fn"],
)
def test_side_rejects_noncallable_adapter_operations_before_sampling(malformed: str) -> None:
    sample_calls = 0

    def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
        nonlocal sample_calls
        del conditions, seed
        sample_calls += 1
        return torch.zeros((batch_size, 1, 4, 4))

    def q_sample(clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        del timesteps, noise
        return clean

    def guided_sample(batch_size: int, labels: torch.Tensor, gradient_fn: Any, seed: int) -> torch.Tensor:
        del labels, gradient_fn, seed
        return torch.zeros((batch_size, 1, 4, 4))

    operations: dict[str, Any] = {
        "sample_fn": sample,
        "q_sample_fn": q_sample,
        "guided_sample_fn": guided_sample,
        "classifier_timestep_fn": None,
    }
    operations[malformed] = "not-callable"
    adapter = CallableDiffusionAdapter(image_shape=(1, 4, 4), num_timesteps=10, **operations)
    attack = AttackSIDEExtraction(
        adapter,
        TwoFeatureExtractor(),
        {
            "authorized_audit": True,
            "synthetic_samples": 4,
            "clusters": 2,
            "min_cluster_size": 1,
            "cohesion_threshold": -1.0,
        },
        audit_fingerprint=f"malformed-{malformed}",
    )

    with pytest.raises(ValueError, match=malformed):
        attack.prepare_attack()

    assert sample_calls == 0


def test_side_provenance_discloses_small_dpm_engineering_choices() -> None:
    adapter = CallableDiffusionAdapter(
        image_shape=(1, 4, 4),
        sample_fn=lambda batch_size, conditions, seed: torch.zeros((batch_size, 1, 4, 4)),
    )
    attack = AttackSIDEExtraction(
        adapter,
        TwoFeatureExtractor(),
        {"authorized_audit": True},
        audit_fingerprint="side-provenance-test",
    )

    scope = attack.description()["scope"]

    assert "Classifier epochs" in scope
    assert "ResNet" in scope
    assert "LoRA branch is excluded" in scope


def test_side_guidance_reconciles_classifier_and_sampler_dtypes() -> None:
    classifier = TinyTimeClassifier(1, 2).double()
    attack = AttackSIDEExtraction(
        CallableDiffusionAdapter(
            image_shape=(1, 4, 4),
            sample_fn=lambda batch_size, conditions, seed: torch.zeros((batch_size, 1, 4, 4)),
        ),
        TwoFeatureExtractor(),
        {"authorized_audit": True, "compute_device": "cpu", "guidance_scale": 1.0},
        audit_fingerprint="guidance-dtype-test",
        classifier=classifier,
    )
    sampler_images = torch.zeros((2, 1, 4, 4), dtype=torch.float32)

    gradient = attack._condition_gradient(  # noqa: SLF001 - direct device/dtype boundary regression
        sampler_images,
        torch.ones(2, dtype=torch.long),
        torch.tensor([0, 1]),
    )

    assert gradient.device == sampler_images.device
    assert gradient.dtype == sampler_images.dtype
    assert torch.isfinite(gradient).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the cross-device regression.")
def test_side_guidance_returns_accelerator_gradient_to_cpu_sampler() -> None:
    classifier = TinyTimeClassifier(1, 2).to("cuda")
    attack = AttackSIDEExtraction(
        CallableDiffusionAdapter(
            image_shape=(1, 4, 4),
            sample_fn=lambda batch_size, conditions, seed: torch.zeros((batch_size, 1, 4, 4)),
        ),
        TwoFeatureExtractor(),
        {"authorized_audit": True, "compute_device": "cuda", "guidance_scale": 1.0},
        audit_fingerprint="guidance-cross-device-test",
        classifier=classifier,
    )
    sampler_images = torch.zeros((2, 1, 4, 4), device="cpu")

    gradient = attack._condition_gradient(  # noqa: SLF001 - direct cross-device boundary regression
        sampler_images,
        torch.ones(2, dtype=torch.long),
        torch.tensor([0, 1]),
    )

    assert gradient.device.type == "cpu"
    assert torch.isfinite(gradient).all()


def test_side_l2_bands_use_configured_minus_one_one_coordinates() -> None:
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
            "l2_bands": {"paper_range": {"lower": 1.35, "upper": 1.4}},
        },
        audit_fingerprint="side-minus-one-one-range",
        reference_images=torch.full((1, 1, 4, 4), -1.0),
        classifier_factory=TinyTimeClassifier,
    )

    attack.prepare_attack()
    result = attack.run_attack()

    assert result.metrics["nearest_l2_mean"] == pytest.approx(1.38)
    assert result.metrics["l2_coordinate_range"] == "minus_one_one"
    assert result.metrics["l2_band_scores"]["paper_range"]["ams"] == 1.0
    assert result.metrics["l2_band_scores"]["paper_range"]["ums"] == 1.0
    torch.testing.assert_close(result.images, torch.full_like(result.images, 0.69))
