"""Public-path integration and execution-trace tests for extraction attacks."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import nn

from leakpro import AbstractExtractionInputHandler, LeakPro
from leakpro.attacks.extraction_attacks.adapters import CallableDiffusionAdapter
from leakpro.schemas import ExtractionTargetConfig, LeakProConfig, TargetConfig


def test_public_extraction_import_does_not_require_non_extraction_image_or_text_stacks() -> None:
    script = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torchvision', 'transformers'}:
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from leakpro import AbstractExtractionInputHandler, LeakPro
assert AbstractExtractionInputHandler is not None
assert LeakPro is not None
"""

    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


class TinyFeatureExtractor(nn.Module):
    """Separate constant dark and bright images in a two-dimensional space."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        means = images.mean(dim=(1, 2, 3))
        return torch.stack((means, 1.0 - means), dim=1)


class TinyTimeClassifier(nn.Module):
    """Small differentiable classifier used only by the integration test."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.image = nn.Linear(in_channels * 4 * 4, num_classes)
        self.time = nn.Linear(1, num_classes, bias=False)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.image(images.flatten(start_dim=1)) + self.time(timesteps.float().unsqueeze(1) / 10.0)


def _alternating_samples(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
    del conditions, seed
    values = torch.arange(batch_size).remainder(2).float()
    return values.view(-1, 1, 1, 1).expand(-1, 1, 4, 4).clone()


class CarliniProvider(AbstractExtractionInputHandler):
    """Conditional black-box provider for the public scheduler path."""

    def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
        def repeated_samples(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
            del seed
            assert conditions == ["memorized"] * batch_size
            return torch.zeros((batch_size, 1, 4, 4))

        return CallableDiffusionAdapter(image_shape=(1, 4, 4), sample_fn=repeated_samples)

    def get_extraction_conditions(self) -> list[str]:
        return ["memorized"]

    def get_extraction_reference_images(self) -> torch.Tensor:
        return torch.zeros((1, 1, 4, 4))


class SIDEProvider(AbstractExtractionInputHandler):
    """White-box toy diffusion provider for the public scheduler path."""

    def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
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
            gradient = gradient_fn(noisy, torch.ones(batch_size, dtype=torch.long), labels)
            assert gradient.shape == noisy.shape
            return _alternating_samples(batch_size, None, 0)

        return CallableDiffusionAdapter(
            image_shape=(1, 4, 4),
            sample_fn=_alternating_samples,
            num_timesteps=10,
            q_sample_fn=q_sample,
            guided_sample_fn=guided_sample,
        )

    def get_extraction_reference_images(self) -> torch.Tensor:
        return torch.stack((torch.zeros((1, 4, 4)), torch.ones((1, 4, 4))))

    def get_side_feature_extractor(self) -> nn.Module:
        return TinyFeatureExtractor()

    def get_side_classifier_factory(self) -> Any:
        return TinyTimeClassifier


def _write_config(
    tmp_path: Any,
    attack: str,
    attack_config: dict[str, Any],
    *,
    target_fingerprint: str = "toy-diffusion-v1",
    config_name: str = "audit.yaml",
    output_dir: Any = None,
) -> str:
    config = {
        "audit": {
            "attack_type": "extraction",
            "attack_list": [{"attack": attack, **attack_config}],
            "data_modality": "image",
            "output_dir": str(output_dir or (tmp_path / "output")),
        },
        "target": {"name": "toy-diffusion", "fingerprint": target_fingerprint},
    }
    config_path = tmp_path / config_name
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(config_path)


def test_extraction_schema_does_not_change_standard_target_parsing(tmp_path: Any) -> None:
    common_audit = {
        "attack_list": [{"attack": "rmia"}],
        "data_modality": "image",
        "output_dir": str(tmp_path),
    }
    extraction = LeakProConfig(
        audit={**common_audit, "attack_type": "extraction"},
        target={"name": "generator", "fingerprint": "generator-v1"},
    )
    mia = LeakProConfig(
        audit={**common_audit, "attack_type": "mia"},
        target={
            "module_path": "target.py",
            "model_class": "Target",
            "target_folder": "target",
            "data_path": "population.pkl",
        },
    )

    assert isinstance(extraction.target, ExtractionTargetConfig)
    assert isinstance(mia.target, TargetConfig)

    with pytest.raises(ValueError, match="fingerprint"):
        LeakProConfig(
            audit={**common_audit, "attack_type": "extraction"},
            target={"name": "generator", "fingerprint": "   "},
        )


@pytest.mark.parametrize("modality", ["text", "tabular", "graph", "timeseries"])
def test_extraction_schema_rejects_non_image_modalities(tmp_path: Any, modality: str) -> None:
    with pytest.raises(ValueError, match="image"):
        LeakProConfig(
            audit={
                "attack_type": "extraction",
                "attack_list": [{"attack": "carlini_diffusion"}],
                "data_modality": modality,
                "output_dir": str(tmp_path),
            },
            target={"name": "generator", "fingerprint": "generator-v1"},
        )


def test_carlini_public_path_persists_trace(tmp_path: Any) -> None:
    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {
            "authorized_audit": True,
            "mode": "conditional_black_box",
            "num_generations_per_condition": 4,
            "generation_batch_size": 2,
            "tile_grid": [2, 2],
            "tiled_l2_threshold": 0.01,
            "min_clique_size": 4,
        },
    )

    result = LeakPro(CarliniProvider, config_path).run_audit()[0]

    assert result.metrics["candidate_count"] == 1
    assert result.metrics["conditions_audited"] == 1
    assert [event["phase"] for event in result.execution_trace] == [
        "prepared",
        "condition_complete",
        "run_complete",
    ]
    assert result.execution_trace[-1]["sampling_calls"] == 2
    assert (tmp_path / "output" / "results" / result.id / "result.json").exists()
    assert (tmp_path / "output" / "results" / result.id / "candidates.npz").exists()


def test_public_path_rejects_a_scalar_string_condition_before_adapter_access(tmp_path: Any) -> None:
    class ScalarStringProvider(AbstractExtractionInputHandler):
        adapter_requested = False

        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            type(self).adapter_requested = True
            return CallableDiffusionAdapter(image_shape=(1, 4, 4), sample_fn=_alternating_samples)

        def get_extraction_conditions(self) -> Any:
            return "cat"

    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {
            "authorized_audit": True,
            "num_generations_per_condition": 2,
            "min_clique_size": 2,
        },
    )

    with pytest.raises(TypeError, match="ordered sequence"):
        LeakPro(ScalarStringProvider, config_path)

    assert ScalarStringProvider.adapter_requested is False


def test_side_public_path_records_training_and_guidance(tmp_path: Any) -> None:
    config_path = _write_config(
        tmp_path,
        "side",
        {
            "authorized_audit": True,
            "compute_device": "cpu",
            "synthetic_samples": 8,
            "synthetic_batch_size": 4,
            "clusters": 2,
            "cohesion_threshold": 0.9,
            "min_cluster_size": 2,
            "classifier_epochs": 1,
            "classifier_batch_size": 4,
            "classifier_learning_rate": 0.001,
            "num_generations": 4,
            "generation_batch_size": 2,
            "guidance_scale": 1.0,
        },
    )

    result = LeakPro(SIDEProvider, config_path).run_audit()[0]
    phases = [event["phase"] for event in result.execution_trace]

    assert phases == [
        "synthetic_dataset_complete",
        "clustering_complete",
        "classifier_training_complete",
        "prepared",
        "run_complete",
    ]
    assert result.execution_trace[-1]["sampling_calls"] == 4
    assert result.execution_trace[-1]["guidance_calls"] == 2
    assert result.metrics["retained_clusters"] == 2


@pytest.mark.parametrize(("classifier_batch_size", "expected_singletons"), [(1, 3), (2, 1)])
def test_side_public_path_trains_singleton_batches_without_dropping_them(
    tmp_path: Any,
    classifier_batch_size: int,
    expected_singletons: int,
) -> None:
    class DefaultClassifierProvider(SIDEProvider):
        def get_side_classifier_factory(self) -> None:
            return None

    config_path = _write_config(
        tmp_path,
        "side",
        {
            "random_seed": 42,
            "authorized_audit": True,
            "compute_device": "cpu",
            "synthetic_samples": 3,
            "synthetic_batch_size": 3,
            "clusters": 2,
            "cohesion_threshold": -1.0,
            "min_cluster_size": 1,
            "classifier_epochs": 1,
            "classifier_batch_size": classifier_batch_size,
            "classifier_base_width": 4,
            "classifier_blocks": [1, 1, 1, 1],
            "timestep_embedding_dim": 8,
            "num_generations": 2,
            "generation_batch_size": 2,
            "guidance_scale": 1.0,
        },
    )

    result = LeakPro(DefaultClassifierProvider, config_path).run_audit()[0]
    training_event = next(
        event for event in result.execution_trace if event["phase"] == "classifier_training_complete"
    )

    assert training_event["singleton_batches"] == expected_singletons
    assert len(result.metrics["classifier_epoch_losses"]) == 1
    assert torch.isfinite(torch.tensor(result.metrics["classifier_epoch_losses"])).all()
    assert result.metrics["images_generated"] == 2


def test_side_classifier_factory_is_seeded_independently_of_ambient_rng(tmp_path: Any) -> None:
    attack_config = {
        "random_seed": 42,
        "authorized_audit": True,
        "compute_device": "cpu",
        "synthetic_samples": 8,
        "synthetic_batch_size": 4,
        "clusters": 2,
        "cohesion_threshold": 0.9,
        "min_cluster_size": 2,
        "classifier_epochs": 1,
        "classifier_batch_size": 4,
        "classifier_learning_rate": 0.001,
        "num_generations": 4,
        "generation_batch_size": 2,
        "guidance_scale": 1.0,
    }
    first_config = _write_config(
        tmp_path,
        "side",
        attack_config,
        config_name="side-first.yaml",
        output_dir=tmp_path / "first-output",
    )
    second_config = _write_config(
        tmp_path,
        "side",
        attack_config,
        config_name="side-second.yaml",
        output_dir=tmp_path / "second-output",
    )

    torch.manual_seed(1)
    first = LeakPro(SIDEProvider, first_config).run_audit()[0]
    torch.manual_seed(999)
    second = LeakPro(SIDEProvider, second_config).run_audit()[0]

    assert first.id == second.id
    assert first.metrics["classifier_epoch_losses"] == second.metrics["classifier_epoch_losses"]
    torch.testing.assert_close(first.images, second.images, rtol=0, atol=0)


def test_public_path_rejects_nonfinite_generated_images(tmp_path: Any) -> None:
    class NonfiniteProvider(AbstractExtractionInputHandler):
        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            return CallableDiffusionAdapter(
                image_shape=(1, 4, 4),
                sample_fn=lambda batch_size, conditions, seed: torch.full((batch_size, 1, 4, 4), torch.nan),
            )

        def get_extraction_conditions(self) -> list[str]:
            return ["fault"]

    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {
            "authorized_audit": True,
            "num_generations_per_condition": 2,
            "generation_batch_size": 2,
            "tile_grid": [2, 2],
            "min_clique_size": 2,
        },
    )

    with pytest.raises(ValueError, match="NaN or infinity"):
        LeakPro(NonfiniteProvider, config_path).run_audit()


@pytest.mark.parametrize(
    ("attack", "attack_config", "provider", "message"),
    [
        (
            "carlini_diffusion",
            {
                "authorized_audit": True,
                "num_generations_per_condition": 2,
                "min_clique_size": 3,
            },
            CarliniProvider,
            "min_clique_size",
        ),
        ("unknown", {"authorized_audit": True}, CarliniProvider, "Unknown extraction attack"),
        ("side", {"authorized_audit": True}, CarliniProvider, "feature_extractor"),
    ],
)
def test_invalid_extraction_configurations_fail_during_construction(
    tmp_path: Any,
    attack: str,
    attack_config: dict[str, Any],
    provider: type[AbstractExtractionInputHandler],
    message: str,
) -> None:
    config_path = _write_config(tmp_path, attack, attack_config)

    with pytest.raises(ValueError, match=message):
        LeakPro(provider, config_path)


@pytest.mark.parametrize(
    ("attack", "attack_config", "expected_exception"),
    [
        ("carlini_diffusion", {"authorized_audit": False}, PermissionError),
        (
            "carlini_diffusion",
            {"authorized_audit": True, "num_generations_per_condition": 2, "min_clique_size": 3},
            ValueError,
        ),
        (
            "carlini_diffusion",
            {"authorized_audit": True, "tiled_l2_threshold": float("inf")},
            ValueError,
        ),
        (
            "carlini_diffusion",
            {"authorized_audit": True, "distance_device": "not-a-device"},
            ValueError,
        ),
        ("side", {"authorized_audit": False}, PermissionError),
        ("side", {"authorized_audit": True, "synthetic_samples": 2, "clusters": 3}, ValueError),
        (
            "side",
            {
                "authorized_audit": True,
                "synthetic_samples": 3,
                "clusters": 2,
                "min_cluster_size": 2,
            },
            ValueError,
        ),
        (
            "side",
            {
                "authorized_audit": True,
                "l2_bands": {"invalid": {"lower": float("nan"), "upper": 1.0}},
            },
            ValueError,
        ),
        (
            "side",
            {"authorized_audit": True, "compute_device": "not-a-device"},
            ValueError,
        ),
    ],
)
def test_authorization_and_config_validation_precede_provider_access(
    tmp_path: Any,
    attack: str,
    attack_config: dict[str, Any],
    expected_exception: type[Exception],
) -> None:
    calls: list[str] = []

    class GuardedProvider(AbstractExtractionInputHandler):
        def __init__(self) -> None:
            calls.append("constructor")

        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            calls.append("adapter")
            return CallableDiffusionAdapter(image_shape=(1, 4, 4), sample_fn=_alternating_samples)

        def get_extraction_conditions(self) -> list[str]:
            calls.append("conditions")
            return ["private"]

        def get_extraction_reference_images(self) -> torch.Tensor:
            calls.append("references")
            return torch.zeros((1, 1, 4, 4))

        def get_side_feature_extractor(self) -> nn.Module:
            calls.append("features")
            return TinyFeatureExtractor()

    config_path = _write_config(tmp_path, attack, attack_config)

    with pytest.raises(expected_exception):
        LeakPro(GuardedProvider, config_path)
    assert calls == []


@pytest.mark.parametrize(("attack", "provider_base"), [("carlini_diffusion", CarliniProvider), ("side", SIDEProvider)])
@pytest.mark.parametrize("delta", [-1, 1])
def test_public_path_rejects_wrong_adapter_batch_count(
    tmp_path: Any,
    attack: str,
    provider_base: type[AbstractExtractionInputHandler],
    delta: int,
) -> None:
    class WrongCountProvider(provider_base):  # type: ignore[valid-type, misc]
        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            base = super().get_diffusion_adapter()

            def wrong_count(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
                del conditions, seed
                return torch.zeros((batch_size + delta, *base.image_shape))

            return CallableDiffusionAdapter(
                image_shape=base.image_shape,
                sample_fn=wrong_count,
                num_timesteps=base.num_timesteps,
                q_sample_fn=base.q_sample_fn,
                guided_sample_fn=base.guided_sample_fn,
                classifier_timestep_fn=base.classifier_timestep_fn,
            )

    attack_config = (
        {
            "authorized_audit": True,
            "num_generations_per_condition": 4,
            "generation_batch_size": 4,
            "tile_grid": [2, 2],
            "min_clique_size": 2,
        }
        if attack == "carlini_diffusion"
        else {
            "authorized_audit": True,
            "compute_device": "cpu",
            "synthetic_samples": 4,
            "synthetic_batch_size": 4,
            "clusters": 2,
            "min_cluster_size": 1,
            "cohesion_threshold": -1.0,
            "classifier_epochs": 1,
            "classifier_batch_size": 2,
            "num_generations": 2,
            "generation_batch_size": 2,
        }
    )
    config_path = _write_config(tmp_path, attack, attack_config)

    expected_actual = 4 + delta
    with pytest.raises(ValueError, match=rf"Requested 4 images, but the adapter returned {expected_actual}"):
        LeakPro(WrongCountProvider, config_path).run_audit()


def test_public_side_path_rejects_wrong_guided_batch_count(tmp_path: Any) -> None:
    class WrongGuidedCountProvider(SIDEProvider):
        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            base = super().get_diffusion_adapter()

            def wrong_guided_count(
                batch_size: int,
                labels: torch.Tensor,
                gradient_fn: Any,
                seed: int,
            ) -> torch.Tensor:
                del seed
                noisy = torch.full((batch_size, *base.image_shape), 0.5)
                gradient_fn(noisy, torch.ones(batch_size, dtype=torch.long), labels)
                return torch.zeros((batch_size + 1, *base.image_shape))

            return CallableDiffusionAdapter(
                image_shape=base.image_shape,
                sample_fn=base.sample_fn,
                num_timesteps=base.num_timesteps,
                q_sample_fn=base.q_sample_fn,
                guided_sample_fn=wrong_guided_count,
                classifier_timestep_fn=base.classifier_timestep_fn,
            )

    config_path = _write_config(
        tmp_path,
        "side",
        {
            "authorized_audit": True,
            "compute_device": "cpu",
            "synthetic_samples": 4,
            "synthetic_batch_size": 4,
            "clusters": 2,
            "min_cluster_size": 1,
            "cohesion_threshold": -1.0,
            "classifier_epochs": 1,
            "classifier_batch_size": 2,
            "num_generations": 2,
            "generation_batch_size": 2,
        },
    )

    with pytest.raises(ValueError, match="Requested 2 images, but the adapter returned 3"):
        LeakPro(WrongGuidedCountProvider, config_path).run_audit()


def test_public_side_path_rejects_ignored_guidance_without_persisting(tmp_path: Any) -> None:
    guided_calls: list[int] = []

    class IgnoredGuidanceProvider(SIDEProvider):
        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            base = super().get_diffusion_adapter()

            def unguided_sample(
                batch_size: int,
                labels: torch.Tensor,
                gradient_fn: Any,
                seed: int,
            ) -> torch.Tensor:
                del labels, gradient_fn, seed
                guided_calls.append(batch_size)
                return torch.zeros((batch_size, *base.image_shape))

            return CallableDiffusionAdapter(
                image_shape=base.image_shape,
                sample_fn=base.sample_fn,
                num_timesteps=base.num_timesteps,
                q_sample_fn=base.q_sample_fn,
                guided_sample_fn=unguided_sample,
                classifier_timestep_fn=base.classifier_timestep_fn,
            )

    config_path = _write_config(
        tmp_path,
        "side",
        {
            "authorized_audit": True,
            "compute_device": "cpu",
            "synthetic_samples": 4,
            "synthetic_batch_size": 4,
            "clusters": 2,
            "min_cluster_size": 1,
            "cohesion_threshold": -1.0,
            "classifier_epochs": 1,
            "classifier_batch_size": 2,
            "num_generations": 2,
            "generation_batch_size": 2,
        },
    )
    audit = LeakPro(IgnoredGuidanceProvider, config_path)

    with pytest.raises(RuntimeError, match="did not invoke"):
        audit.run_audit()
    assert guided_calls == [2]

    with pytest.raises(RuntimeError, match="one-shot"):
        audit.run_audit()
    assert guided_calls == [2]

    assert not any((tmp_path / "output" / "results").iterdir())
    data_objects = tmp_path / "output" / "data_objects"
    assert not data_objects.exists() or not any(data_objects.iterdir())


def test_result_identity_separates_target_fingerprints(tmp_path: Any) -> None:
    output_dir = tmp_path / "shared-output"
    attack_config = {
        "authorized_audit": True,
        "num_generations_per_condition": 4,
        "generation_batch_size": 4,
        "tile_grid": [2, 2],
        "min_clique_size": 4,
    }
    first_config = _write_config(
        tmp_path,
        "carlini_diffusion",
        attack_config,
        target_fingerprint="checkpoint-a",
        config_name="audit-a.yaml",
        output_dir=output_dir,
    )
    second_config = _write_config(
        tmp_path,
        "carlini_diffusion",
        attack_config,
        target_fingerprint="checkpoint-b",
        config_name="audit-b.yaml",
        output_dir=output_dir,
    )

    first = LeakPro(CarliniProvider, first_config).run_audit()[0]
    second = LeakPro(CarliniProvider, second_config).run_audit()[0]

    assert first.id != second.id
    assert (output_dir / "results" / first.id / "result.json").exists()
    assert (output_dir / "results" / second.id / "result.json").exists()
    assert first.provenance["audit_fingerprint"] != second.provenance["audit_fingerprint"]


def test_duplicate_result_requires_explicit_overwrite(tmp_path: Any) -> None:
    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {
            "authorized_audit": True,
            "num_generations_per_condition": 4,
            "generation_batch_size": 4,
            "tile_grid": [2, 2],
            "min_clique_size": 4,
        },
    )
    LeakPro(CarliniProvider, config_path).run_audit()

    with pytest.raises(FileExistsError, match="overwrite_results"):
        LeakPro(CarliniProvider, config_path).run_audit()


def test_duplicate_result_is_rejected_before_sampling(tmp_path: Any) -> None:
    sample_calls: list[int] = []

    class RecordingProvider(CarliniProvider):
        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
                del conditions, seed
                sample_calls.append(batch_size)
                return torch.zeros((batch_size, 1, 4, 4))

            return CallableDiffusionAdapter(image_shape=(1, 4, 4), sample_fn=sample)

    attack_config = {
        "authorized_audit": True,
        "num_generations_per_condition": 4,
        "generation_batch_size": 4,
        "tile_grid": [2, 2],
        "min_clique_size": 4,
    }
    config_path = _write_config(tmp_path, "carlini_diffusion", attack_config)
    LeakPro(CarliniProvider, config_path).run_audit()

    with pytest.raises(FileExistsError, match="overwrite_results"):
        LeakPro(RecordingProvider, config_path).run_audit()
    assert sample_calls == []


def test_duplicate_result_ids_inside_one_audit_are_rejected_before_sampling(tmp_path: Any) -> None:
    sample_calls: list[int] = []

    class RecordingProvider(CarliniProvider):
        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
                del conditions, seed
                sample_calls.append(batch_size)
                return torch.zeros((batch_size, 1, 4, 4))

            return CallableDiffusionAdapter(image_shape=(1, 4, 4), sample_fn=sample)

    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {
            "authorized_audit": True,
            "num_generations_per_condition": 4,
            "generation_batch_size": 4,
            "tile_grid": [2, 2],
            "min_clique_size": 4,
        },
    )
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    payload["audit"]["attack_list"] *= 2
    Path(config_path).write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate extraction result IDs"):
        LeakPro(RecordingProvider, config_path).run_audit()
    assert sample_calls == []


def test_overwrite_flag_does_not_change_identity_and_permits_replacement(tmp_path: Any) -> None:
    attack_config = {
        "authorized_audit": True,
        "num_generations_per_condition": 4,
        "generation_batch_size": 4,
        "tile_grid": [2, 2],
        "min_clique_size": 4,
    }
    initial_config = _write_config(
        tmp_path,
        "carlini_diffusion",
        attack_config,
        config_name="initial.yaml",
    )
    initial = LeakPro(CarliniProvider, initial_config).run_audit()[0]

    overwrite_config = _write_config(
        tmp_path,
        "carlini_diffusion",
        {**attack_config, "overwrite_results": True},
        config_name="overwrite.yaml",
    )
    replacement = LeakPro(CarliniProvider, overwrite_config).run_audit()[0]

    assert replacement.id == initial.id
    persisted = tmp_path / "output" / "results" / initial.id / "result.json"
    assert json.loads(persisted.read_text(encoding="utf-8"))["config"]["overwrite_results"] is True


def test_same_instance_replay_fails_before_sampling_and_preserves_bundle(tmp_path: Any) -> None:
    sample_calls: list[tuple[str, int]] = []

    class RecordingSIDEProvider(SIDEProvider):
        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
                sample_calls.append(("sample", batch_size))
                return _alternating_samples(batch_size, conditions, seed)

            def q_sample(clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
                scale = timesteps.float().view(-1, 1, 1, 1) / 100.0
                return clean + scale * noise

            def guided_sample(
                batch_size: int,
                labels: torch.Tensor,
                gradient_fn: Any,
                seed: int,
            ) -> torch.Tensor:
                sample_calls.append(("guided", batch_size))
                noisy = torch.full((batch_size, 1, 4, 4), 0.5)
                gradient_fn(noisy, torch.ones(batch_size, dtype=torch.long), labels)
                return _alternating_samples(batch_size, None, seed)

            return CallableDiffusionAdapter(
                image_shape=(1, 4, 4),
                sample_fn=sample,
                num_timesteps=10,
                q_sample_fn=q_sample,
                guided_sample_fn=guided_sample,
            )

    config_path = _write_config(
        tmp_path,
        "side",
        {
            "authorized_audit": True,
            "overwrite_results": True,
            "compute_device": "cpu",
            "synthetic_samples": 4,
            "synthetic_batch_size": 4,
            "clusters": 2,
            "min_cluster_size": 1,
            "cohesion_threshold": -1.0,
            "classifier_epochs": 1,
            "classifier_batch_size": 2,
            "num_generations": 2,
            "generation_batch_size": 2,
        },
    )
    audit = LeakPro(RecordingSIDEProvider, config_path)
    first = audit.run_audit()[0]
    persisted = tmp_path / "output" / "results" / first.id / "result.json"
    first_bundle = persisted.read_bytes()
    first_call_count = len(sample_calls)

    with pytest.raises(RuntimeError, match="one-shot"):
        audit.run_audit()

    assert len(sample_calls) == first_call_count
    assert persisted.read_bytes() == first_bundle


def test_conditions_are_fingerprinted_not_persisted(tmp_path: Any) -> None:
    secret = "private-prompt-marker-7f42"

    class SecretProvider(CarliniProvider):
        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            return CallableDiffusionAdapter(
                image_shape=(1, 4, 4),
                sample_fn=lambda batch_size, conditions, seed: torch.zeros((batch_size, 1, 4, 4)),
            )

        def get_extraction_conditions(self) -> list[str]:
            return [secret]

    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {
            "authorized_audit": True,
            "num_generations_per_condition": 4,
            "generation_batch_size": 4,
            "tile_grid": [2, 2],
            "min_clique_size": 4,
        },
    )
    LeakPro(SecretProvider, config_path).run_audit()

    persisted_json = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "output").rglob("*.json"))
    assert secret not in persisted_json
    assert "condition_fingerprint" in persisted_json


def test_public_path_rejects_noncanonical_condition_objects(tmp_path: Any) -> None:
    class UnsupportedCondition:
        pass

    class UnsupportedConditionProvider(CarliniProvider):
        def get_extraction_conditions(self) -> list[object]:
            return [UnsupportedCondition()]

    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {"authorized_audit": True},
    )

    with pytest.raises(TypeError, match="Unsupported extraction condition type"):
        LeakPro(UnsupportedConditionProvider, config_path)


def test_extraction_pdf_is_rejected_before_attack_execution(tmp_path: Any) -> None:
    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {
            "authorized_audit": True,
            "num_generations_per_condition": 4,
            "tile_grid": [2, 2],
            "min_clique_size": 4,
        },
    )
    audit = LeakPro(CarliniProvider, config_path)

    with pytest.raises(NotImplementedError, match="PDF reporting"):
        audit.run_audit(create_pdf=True)

    assert not any((tmp_path / "output" / "results").iterdir())


def test_extraction_requires_explicit_handler_type(tmp_path: Any) -> None:
    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {"authorized_audit": True, "num_generations_per_condition": 2, "min_clique_size": 2},
    )

    with pytest.raises(TypeError, match="AbstractExtractionInputHandler"):
        LeakPro(object, config_path)
