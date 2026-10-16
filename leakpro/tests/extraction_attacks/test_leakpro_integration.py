"""Public-path integration tests for extraction attacks."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from examples.extraction.cifar10.cifar10_handler import CIFAR10ExtractionHandler
from leakpro import AbstractExtractionInputHandler, LeakPro
from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.attacks.extraction_attacks.adapters import DiffusionAdapter
from leakpro.attacks.extraction_attacks.protocols import ExtractionAdapter
from leakpro.tests.extraction_attacks.image_sampler import image_sampler
from leakpro.attacks.extraction_attacks.attack_factory_extraction import AttackFactoryExtraction
from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction
from leakpro.attacks.utils.diff_mi.script_util import create_gaussian_diffusion
from leakpro.schemas import ExtractionTargetConfig, LeakProConfig, TargetConfig


@pytest.mark.parametrize("attack_type", ["mia", "gia", "minv", "extraction"])
def test_scheduler_skips_failed_creation_without_mislabeling_survivors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, attack_type: str,
) -> None:
    result = SimpleNamespace(save=Mock())
    attack = SimpleNamespace(
        result_id="valid-result", config=SimpleNamespace(overwrite_results=False), optuna_params=0,
        prepare_attack=Mock(), run_attack=Mock(return_value=result),
    )
    factory = SimpleNamespace(
        attack_classes={"broken": object, "valid": object},
        create_attack=Mock(side_effect=[ValueError("provider unavailable"), attack]),
    )
    monkeypatch.setattr(AttackScheduler, "_initialize_factory", lambda self, name: setattr(self, "attack_factory", factory))
    handler = SimpleNamespace(configs=SimpleNamespace(audit=SimpleNamespace(
        attack_type=attack_type, attack_list=[{"attack": "broken"}, {"attack": "valid", "random_seed": 7}],
    )))

    scheduler = AttackScheduler(handler, str(tmp_path))
    results = scheduler.run_attacks()

    assert scheduler.attack_names == ["valid"]
    assert scheduler.attack_configs == [{"random_seed": 7}]
    assert results == [{"attack_type": "valid", "attack_object": attack, "result_object": result}]
    assert "Failed to create attack: broken" in caplog.text
    attack.prepare_attack.assert_called_once_with()
    attack.run_attack.assert_called_once_with()
    result.save.assert_called_once_with(attack_obj=attack, output_dir=str(tmp_path))


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

    def get_diffusion_adapter(self) -> ExtractionAdapter[torch.Tensor]:
        def repeated_samples(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
            del seed
            assert conditions == ["memorized"] * batch_size
            return torch.zeros((batch_size, 1, 4, 4))

        return image_sampler(image_shape=(1, 4, 4), sample_fn=repeated_samples, model=nn.Identity())

    def get_extraction_conditions(self) -> list[str]:
        return ["memorized"]

    def get_extraction_reference_images(self) -> torch.Tensor:
        return torch.zeros((1, 1, 4, 4))


class SIDEProvider(AbstractExtractionInputHandler):
    """White-box toy diffusion provider for the public scheduler path."""

    def get_diffusion_adapter(self) -> ExtractionAdapter[torch.Tensor]:
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

        return image_sampler(
            image_shape=(1, 4, 4),
            sample_fn=_alternating_samples,
            num_timesteps=10,
            q_sample_fn=q_sample,
            guided_sample_fn=guided_sample,
         model=nn.Identity())

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
    target_name: str = "toy-diffusion",
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
        "target": {"name": target_name},
    }
    config_path = tmp_path / config_name
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(config_path)


def test_public_clique_budget_failure_publishes_no_partial_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    settings = {"num_generations_per_condition": 4, "min_clique_size": 2, "max_clique_search_nodes": 1}
    config_path = _write_config(tmp_path, "carlini_diffusion", settings)
    audit = LeakPro(CarliniProvider, config_path)
    with pytest.raises(RuntimeError, match="budget of 1 search nodes"):
        audit.run_audit()
    assert "No exact result was returned" in caplog.text
    assert not list((tmp_path / "output").rglob("result.txt"))
    assert not list((tmp_path / "output").rglob("candidates.npz"))

    settings["max_clique_search_nodes"] = 5
    recovered = LeakPro(CarliniProvider, _write_config(tmp_path, "carlini_diffusion", settings))
    assert recovered.attack_scheduler.attacks[0].result_id == audit.attack_scheduler.attacks[0].result_id
    results = recovered.run_audit()
    assert len(results) == 1 and len(results[0].candidates) == 1
    assert len(list((tmp_path / "output").rglob("result.txt"))) == 1


def test_extraction_schema_does_not_change_standard_target_parsing(tmp_path: Any) -> None:
    common_audit = {
        "attack_list": [{"attack": "rmia"}],
        "data_modality": "image",
        "output_dir": str(tmp_path),
    }
    extraction = LeakProConfig(
        audit={**common_audit, "attack_type": "extraction"},
        target={"name": "generator"},
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

    with pytest.raises(ValueError, match="image"):
        LeakProConfig(
            audit={**common_audit, "attack_type": "extraction", "data_modality": "text"},
            target={"name": "generator"},
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
    config_path = Path(_write_config(tmp_path, attack_name, {}))
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


@pytest.mark.parametrize("seed", [-1, 2**32])
@pytest.mark.parametrize("seed_source", ["audit", "attack"])
def test_invalid_side_seed_is_rejected_before_loading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, seed: int, seed_source: str,
) -> None:
    get_features = Mock(side_effect=AssertionError("Invalid seed must fail before loading target inputs"))
    monkeypatch.setattr(SIDEProvider, "get_side_feature_extractor", get_features)
    path = Path(_write_config(tmp_path, "side", {}))
    config = yaml.safe_load(path.read_text())
    destination = config["audit"] if seed_source == "audit" else config["audit"]["attack_list"][0]
    destination["random_seed"] = seed
    path.write_text(yaml.safe_dump(config))

    audit = LeakPro(SIDEProvider, str(path))

    assert audit.attack_scheduler.attacks == []
    assert "random_seed" in caplog.text and "Failed to create attack: side" in caplog.text
    get_features.assert_not_called()


def test_two_attacks_reuse_provider_inputs_without_hashing_references(tmp_path: Path) -> None:
    """Construct both attacks with one target identity and no reference digest."""
    calls = {"adapter": 0, "references": 0}

    class CountingProvider(SIDEProvider):
        def get_diffusion_adapter(self) -> ExtractionAdapter[torch.Tensor]:
            calls["adapter"] += 1
            return super().get_diffusion_adapter()

        def get_extraction_reference_images(self) -> torch.Tensor:
            calls["references"] += 1
            return super().get_extraction_reference_images()

    config_path = Path(_write_config(tmp_path, "side", {}))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["audit"]["attack_list"].insert(
        0,
        {
            "attack": "carlini_diffusion",
            "mode": "unconditional_reference_audit",
            "reference_neighbors": 2,
        },
    )
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    attacks = LeakPro(CountingProvider, str(config_path)).attack_scheduler.attacks

    assert len(attacks) == 2
    assert calls == {"adapter": 1, "references": 1}
    assert all(attack.target_name == "toy-diffusion" for attack in attacks)


def test_factory_constructs_a_registered_subclass_through_the_public_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RegisteredCarlini(AttackCarliniExtraction):
        pass

    class ExampleProvider(CIFAR10ExtractionHandler):
        pass

    ExampleProvider.configure(
        adapter=image_sampler((1, 4, 4), _alternating_samples, model=nn.Identity()),
        references=torch.stack((torch.zeros((1, 4, 4)), torch.ones((1, 4, 4)))),
        feature_extractor=TinyFeatureExtractor(), feature_transform=nn.Identity(),
    )
    monkeypatch.setitem(AttackFactoryExtraction.attack_classes, "registered_carlini", RegisteredCarlini)
    config_path = _write_config(
        tmp_path, "registered_carlini",
        {"mode": "unconditional_reference_audit", "num_unconditional_generations": 2, "reference_neighbors": 2},
    )
    audit = LeakPro(ExampleProvider, config_path)
    result = audit.run_audit()[0]

    assert isinstance(audit.attack_scheduler.attacks[0], RegisteredCarlini)
    assert result.images.shape == (2, 1, 4, 4)
    assert result.provenance["target"] == "toy-diffusion"


@pytest.mark.parametrize("device_type", ["hpu", "mps", "xpu"])
def test_accelerators_are_accepted_as_extraction_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_type: str,
) -> None:
    """The public factory accepts any backend selected by the shared helper."""
    monkeypatch.setattr(
        "leakpro.attacks.extraction_attacks.side.get_device",
        lambda: torch.device(device_type),
    )
    config_path = _write_config(
        tmp_path, "side", {"compute_device": "auto", "distance_device": "auto"},
    )
    attack = LeakPro(SIDEProvider, config_path).attack_scheduler.attacks[0]
    assert attack.device == torch.device(device_type)


def test_carlini_public_path_persists_metrics(tmp_path: Any) -> None:
    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {
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
    assert result.metrics["sampling_calls"] == 2
    result_dir = tmp_path / "output" / "results" / result.id
    metadata_path = tmp_path / "output" / "data_objects" / f"{result.id}.json"
    array_path = result_dir / "candidates.npz"
    assert metadata_path.exists()
    assert array_path.exists()
    assert (result_dir / "result.txt").exists()
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
        def get_diffusion_adapter(self) -> ExtractionAdapter[torch.Tensor]:
            def sample(batch_size: int, conditions: Sequence[Any] | None, seed: int) -> torch.Tensor:
                sample_calls.append(batch_size)
                return torch.zeros((batch_size, 1, 4, 4))

            return image_sampler(image_shape=(1, 4, 4), sample_fn=sample, model=nn.Identity())

    with pytest.raises(FileExistsError, match="overwrite_results"):
        LeakPro(RecordingProvider, config_path).run_audit()
    assert sample_calls == []
    assert saved_bundle == (metadata_path.read_bytes(), array_path.read_bytes())

    different_target = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    different_target["target"]["name"] = "different-target"
    second_config = tmp_path / "different-target.yaml"
    second_config.write_text(yaml.safe_dump(different_target), encoding="utf-8")
    class DifferentWeightsProvider(CarliniProvider):
        def get_diffusion_adapter(self):
            adapter = super().get_diffusion_adapter()
            adapter.model = nn.Linear(1, 1)
            return adapter

    second = LeakPro(DifferentWeightsProvider, str(second_config)).run_audit()[0]
    assert second.id != result.id
    assert (tmp_path / "output" / "data_objects" / f"{second.id}.json").exists()


@pytest.mark.parametrize("seed", [42, 0, 2**32 - 1])
def test_side_public_path_records_training_and_guidance(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, seed: int,
) -> None:
    marked_devices: list[str] = []
    monkeypatch.setattr("leakpro.attacks.extraction_attacks.side.mark_step", lambda device: marked_devices.append(device.type))
    config_path = _write_config(
        tmp_path,
        "side",
        {
            "random_seed": seed,
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
            "overwrite_results": True,
        },
    )

    audit = LeakPro(SIDEProvider, config_path)
    result = audit.run_audit()[0]
    assert marked_devices == ["cpu"] * 4
    assert result.metrics["sampling_calls"] == 4
    assert result.metrics["synthetic_sampling_calls"] == 2
    assert result.metrics["guidance_calls"] == 2
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

    rerun = audit.run_audit()[0]
    assert rerun.metrics == result.metrics
    torch.testing.assert_close(rerun.images, result.images, rtol=0, atol=0)


def test_side_public_path_trains_default_classifier_with_a_singleton_batch(tmp_path: Path) -> None:
    class DefaultClassifierProvider(SIDEProvider):
        def get_side_classifier_factory(self) -> None:
            return None

    config_path = _write_config(
        tmp_path,
        "side",
        {
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
    assert result.metrics["singleton_classifier_batches"] == 1
    assert result.images.shape == (2, 1, 4, 4)
    assert not torch.allclose(result.images, torch.full_like(result.images, 0.5), atol=1e-6, rtol=0)


def test_public_path_rejects_nonfinite_generated_images(tmp_path: Path) -> None:
    class NonfiniteProvider(CarliniProvider):
        def get_diffusion_adapter(self) -> ExtractionAdapter[torch.Tensor]:
            return image_sampler(
                image_shape=(1, 4, 4),
                sample_fn=lambda batch_size, conditions, seed: torch.full((batch_size, 1, 4, 4), torch.nan),
             model=nn.Identity())

    config_path = _write_config(
        tmp_path,
        "carlini_diffusion",
        {"num_generations_per_condition": 2, "min_clique_size": 2},
    )
    with pytest.raises(ValueError, match="NaN or infinity"):
        LeakPro(NonfiniteProvider, config_path).run_audit()
    assert not any((tmp_path / "output" / "results").iterdir())


@pytest.mark.parametrize(
    ("attack", "attack_config"),
    [
        ("carlini_diffusion", {"num_generations_per_condition": 2, "min_clique_size": 3}),
        ("side", {"synthetic_samples": 2, "clusters": 3}),
        ("unknown", {}),
    ],
)
def test_invalid_attack_settings_are_skipped_before_provider_access(
    tmp_path: Any,
    attack: str,
    attack_config: dict[str, Any],
) -> None:
    calls: list[str] = []

    class GuardedProvider(AbstractExtractionInputHandler):
        def __init__(self) -> None:
            calls.append("constructor")

        def get_diffusion_adapter(self) -> ExtractionAdapter[torch.Tensor]:
            calls.append("adapter")
            return image_sampler(image_shape=(1, 4, 4), sample_fn=_alternating_samples, model=nn.Identity())

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

    audit = LeakPro(GuardedProvider, config_path)
    assert audit.attack_scheduler.attacks == []
    assert calls == []


def test_public_side_path_rejects_ignored_guidance_without_persisting(tmp_path: Any) -> None:
    guided_calls: list[int] = []

    class IgnoredGuidanceProvider(SIDEProvider):
        def get_diffusion_adapter(self) -> ExtractionAdapter[torch.Tensor]:
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

            base.sample_with_classifier_guidance = unguided_sample
            return base

    config_path = _write_config(
        tmp_path,
        "side",
        {
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

    with pytest.raises(RuntimeError, match="did not invoke"):
        audit.run_audit()
    assert guided_calls == [2, 2]

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


def test_both_attacks_run_through_the_vendored_unconditional_diffusion(tmp_path: Path) -> None:
    """Exercise forward noising, SIDE gradients and result saving with the shared sampler."""
    sampled: list[tuple[int, torch.Tensor]] = []
    guided_seeds: list[int] = []

    class TracedAdapter(DiffusionAdapter):
        def sample(self, batch_size, *, conditions, seed):
            images = super().sample(batch_size, conditions=conditions, seed=seed)
            sampled.append((seed, images.clone()))
            return images

        def sample_with_classifier_guidance(self, batch_size, *, labels, gradient_fn, seed):
            guided_seeds.append(seed)
            return super().sample_with_classifier_guidance(batch_size, labels=labels, gradient_fn=gradient_fn, seed=seed)

    class UnconditionalDenoiser(nn.Module):
        def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            assert timesteps.shape == (images.shape[0],)
            return torch.cat((images * 0.1, torch.zeros_like(images)), dim=1)

    class SharedProvider(SIDEProvider):
        def get_diffusion_adapter(self) -> DiffusionAdapter:
            diffusion = create_gaussian_diffusion(
                diffusion_steps=6, learn_sigma=True, noise_schedule="cosine", rescale_timesteps=True,
            )
            return TracedAdapter(
                model=UnconditionalDenoiser(), diffusion=diffusion, image_shape=(1, 4, 4),
                sampling_steps=3, clip_guided=True,
            )

    config_path = Path(_write_config(tmp_path, "side", {
        "compute_device": "cpu", "image_range": "minus_one_one",
        "synthetic_samples": 4, "synthetic_batch_size": 2, "clusters": 2,
        "min_cluster_size": 1, "cohesion_threshold": -1.0,
        "classifier_epochs": 1, "classifier_batch_size": 2,
        "num_generations": 4, "generation_batch_size": 2,
    }))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["audit"]["attack_list"].insert(0, {
        "attack": "carlini_diffusion", "mode": "unconditional_reference_audit",
        "image_range": "minus_one_one", "num_unconditional_generations": 4,
        "generation_batch_size": 2, "reference_neighbors": 2,
    })
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    results = LeakPro(SharedProvider, str(config_path)).run_audit()

    assert len(results) == 2
    assert results[0].metrics["sampling_calls"] == 2
    assert results[1].metrics["guidance_calls"] == 6
    assert results[1].metrics["retained_clusters"] == 2
    for result in results:
        assert torch.isfinite(result.images).all()
        assert (tmp_path / "output" / "results" / result.id / "candidates.npz").exists()

    assert len(sampled) == 4
    all_seeds = [seed for seed, _images in sampled] + guided_seeds
    assert len(set(all_seeds)) == len(all_seeds)
    assert not torch.equal(sampled[0][1], sampled[2][1])
    assert not torch.equal(sampled[1][1], sampled[3][1])


@pytest.mark.parametrize("attack_name", ["carlini_diffusion", "side"])
def test_public_result_identity_uses_weights_and_ignores_display_name(tmp_path: Path, attack_name: str) -> None:
    class WeightedProvider(SIDEProvider):
        target_model = nn.Linear(1, 1)

        def get_diffusion_adapter(self):
            adapter = super().get_diffusion_adapter()
            adapter.model = self.target_model
            return adapter

        def get_extraction_conditions(self):
            return ["label"]

    path = _write_config(tmp_path, attack_name, {})
    original = LeakPro(WeightedProvider, path).attack_scheduler.attacks[0].result_id
    renamed_path = _write_config(tmp_path, attack_name, {}, target_name="renamed", config_name="renamed.yaml")
    renamed = LeakPro(WeightedProvider, renamed_path).attack_scheduler.attacks[0].result_id
    assert renamed == original
    with torch.no_grad():
        WeightedProvider.target_model.weight.add_(1)
    changed = LeakPro(WeightedProvider, path).attack_scheduler.attacks[0].result_id
    assert changed != original
