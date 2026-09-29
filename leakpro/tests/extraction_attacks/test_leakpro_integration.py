"""Public-path integration and execution-trace tests for extraction attacks."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from examples.extraction.cifar10.cifar10_handler import CIFAR10ExtractionHandler
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
            return noisy + 0.25 * gradient.tanh()

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
    target_hash: str = "toy-diffusion-v1",
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
        "target": {"name": "toy-diffusion", "hash": target_hash},
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
        target={"name": "generator", "hash": "generator-v1"},
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

    with pytest.raises(ValueError, match="hash"):
        LeakProConfig(
            audit={**common_audit, "attack_type": "extraction"},
            target={"name": "generator", "hash": "   "},
        )
    with pytest.raises(ValueError, match="image"):
        LeakProConfig(
            audit={**common_audit, "attack_type": "extraction", "data_modality": "text"},
            target={"name": "generator", "hash": "generator-v1"},
        )


@pytest.mark.parametrize(
    ("attack_name", "provider"),
    [
        ("carlini_diffusion", CarliniProvider),
        ("side", SIDEProvider),
    ],
)
def test_extraction_attack_seed_inherits_audit_seed_unless_overridden(
    tmp_path: Path,
    attack_name: str,
    provider: type[AbstractExtractionInputHandler],
) -> None:
    """The public audit path applies its seed without replacing an explicit attack seed."""
    config_path = Path(_write_config(tmp_path, attack_name, {"authorized_audit": True}))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["audit"]["random_seed"] = 7
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    inherited = LeakPro(provider, str(config_path)).attack_scheduler.attacks[0]
    assert inherited.config.random_seed == 7

    config["audit"]["attack_list"][0]["random_seed"] = 11
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    overridden = LeakPro(provider, str(config_path)).attack_scheduler.attacks[0]
    assert overridden.config.random_seed == 11
    assert overridden.result_id != inherited.result_id


def test_cifar10_result_id_tracks_model_and_sampling_without_editing_yaml(tmp_path: Path) -> None:
    class ExampleProvider(CIFAR10ExtractionHandler):
        pass

    model = nn.Linear(1, 1)
    feature_extractor = nn.Linear(1, 1)
    adapter = CallableDiffusionAdapter(image_shape=(1, 4, 4), sample_fn=_alternating_samples)
    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {
            "authorized_audit": True,
            "mode": "unconditional_reference_audit",
            "reference_neighbors": 2,
        },
    )

    def result_id(steps: int) -> str:
        ExampleProvider.configure(
            adapter=adapter,
            target_model=model,
            sampling_steps=steps,
            references=torch.zeros((2, 1, 4, 4)),
            feature_extractor=feature_extractor,
            feature_transform=nn.Identity(),
        )
        return LeakPro(ExampleProvider, config_path).attack_scheduler.attacks[0].result_id

    original = result_id(50)
    assert result_id(100) != original
    with torch.no_grad():
        model.weight.add_(1)
    assert result_id(50) != original


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
    result_dir = tmp_path / "output" / "results" / result.id
    metadata_path = result_dir / "result.json"
    array_path = result_dir / "candidates.npz"
    assert metadata_path.exists()
    assert array_path.exists()
    assert not (tmp_path / "output" / "data_objects").exists()
    metadata = metadata_path.read_text(encoding="utf-8")
    assert json.loads(metadata)["candidate_count"] == 1
    assert json.loads(metadata)["candidates"][0]["image_index"] == 0
    with np.load(array_path) as archive:
        np.testing.assert_array_equal(archive["images"], result.images.numpy())
    assert '"memorized"' not in metadata
    assert "condition_hash" in metadata
    saved_bundle = (metadata_path.read_bytes(), array_path.read_bytes())

    sample_calls: list[int] = []

    class RecordingProvider(CarliniProvider):
        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
                sample_calls.append(batch_size)
                return torch.zeros((batch_size, 1, 4, 4))

            return CallableDiffusionAdapter(image_shape=(1, 4, 4), sample_fn=sample)

    with pytest.raises(FileExistsError, match="overwrite_results"):
        LeakPro(RecordingProvider, config_path).run_audit()
    assert sample_calls == []
    assert saved_bundle == (metadata_path.read_bytes(), array_path.read_bytes())

    different_target = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    different_target["target"]["hash"] = "different-checkpoint"
    second_config = tmp_path / "different-target.yaml"
    second_config.write_text(yaml.safe_dump(different_target), encoding="utf-8")
    second = LeakPro(CarliniProvider, str(second_config)).run_audit()[0]
    assert second.id != result.id
    assert (tmp_path / "output" / "results" / second.id / "result.json").exists()


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
    assert not torch.allclose(result.images, torch.full_like(result.images, 0.5), atol=1e-6, rtol=0)

    repeated_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    repeated_config["audit"]["output_dir"] = str(tmp_path / "repeated-output")
    repeated_path = tmp_path / "repeated.yaml"
    repeated_path.write_text(yaml.safe_dump(repeated_config), encoding="utf-8")
    torch.manual_seed(999)
    repeated = LeakPro(SIDEProvider, str(repeated_path)).run_audit()[0]
    assert repeated.id == result.id
    assert repeated.metrics["classifier_epoch_losses"] == result.metrics["classifier_epoch_losses"]
    torch.testing.assert_close(repeated.images, result.images, rtol=0, atol=0)


def test_side_public_path_trains_default_classifier_with_a_singleton_batch(tmp_path: Path) -> None:
    class DefaultClassifierProvider(SIDEProvider):
        def get_side_classifier_factory(self) -> None:
            return None

    config_path = _write_config(
        tmp_path,
        "side",
        {
            "authorized_audit": True,
            "compute_device": "cpu",
            "synthetic_samples": 3,
            "clusters": 2,
            "min_cluster_size": 1,
            "cohesion_threshold": -1.0,
            "classifier_epochs": 1,
            "classifier_batch_size": 2,
            "classifier_base_width": 4,
            "classifier_blocks": [1, 1, 1, 1],
            "timestep_embedding_dim": 8,
            "num_generations": 2,
            "generation_batch_size": 2,
        },
    )

    result = LeakPro(DefaultClassifierProvider, config_path).run_audit()[0]
    training = next(event for event in result.execution_trace if event["phase"] == "classifier_training_complete")
    assert training["singleton_batches"] == 1
    assert result.images.shape == (2, 1, 4, 4)
    assert not torch.allclose(result.images, torch.full_like(result.images, 0.5), atol=1e-6, rtol=0)


def test_public_path_rejects_nonfinite_generated_images(tmp_path: Path) -> None:
    class NonfiniteProvider(CarliniProvider):
        def get_diffusion_adapter(self) -> CallableDiffusionAdapter:
            return CallableDiffusionAdapter(
                image_shape=(1, 4, 4),
                sample_fn=lambda batch_size, conditions, seed: torch.full((batch_size, 1, 4, 4), torch.nan),
            )

    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {"authorized_audit": True, "num_generations_per_condition": 2, "min_clique_size": 2},
    )
    with pytest.raises(ValueError, match="NaN or infinity"):
        LeakPro(NonfiniteProvider, config_path).run_audit()
    assert not any((tmp_path / "output" / "results").iterdir())


@pytest.mark.parametrize(
    ("attack", "attack_config", "expected_exception"),
    [
        ("carlini_diffusion", {"authorized_audit": False}, PermissionError),
        (
            "carlini_diffusion",
            {"authorized_audit": True, "num_generations_per_condition": 2, "min_clique_size": 3},
            ValueError,
        ),
        ("side", {"authorized_audit": False}, PermissionError),
        ("side", {"authorized_audit": True, "synthetic_samples": 2, "clusters": 3}, ValueError),
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


def test_core_import_does_not_load_extraction_attacks() -> None:
    """Extraction remains optional until an extraction API is requested."""
    script = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.startswith('leakpro.attacks.extraction_attacks') or name == 'leakpro.input_handler.extraction_handler':
        raise ImportError('extraction must stay lazy')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from leakpro import LeakPro, AbstractInputHandler
assert LeakPro is not None
assert AbstractInputHandler is not None
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
