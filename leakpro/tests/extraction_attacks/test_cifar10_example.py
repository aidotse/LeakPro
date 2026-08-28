"""Checks for the runnable CIFAR-10 extraction notebook helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

from examples.extraction.cifar10.cifar10_handler import load_audit_config
from examples.extraction.cifar10.cifar10_model import (
    GaussianDiffusion,
    RunProfile,
    SmallTimeUNet,
    make_adapter,
    sha256_mapping,
    sha256_module_state,
    sha256_tensor,
    train_or_load_target,
)
from leakpro.attacks.extraction_attacks.configs import CarliniConfig, SIDEConfig


def _test_profile() -> RunProfile:
    return RunProfile(
        name="test",
        seed=7,
        train_size=8,
        epochs=1,
        train_batch_size=4,
        learning_rate=1e-3,
        timesteps=4,
        sampling_steps=4,
        model_channels=8,
        reference_size=8,
    )


def test_forward_noising_matches_the_closed_form() -> None:
    """The notebook diffusion must expose the SIDE forward process exactly."""
    diffusion = GaussianDiffusion(timesteps=4, sampling_steps=4, device=torch.device("cpu"))
    clean = torch.full((2, 3, 4, 4), 0.25)
    noise = torch.full_like(clean, -0.5)
    timesteps = torch.tensor([0, 3])

    observed = diffusion.q_sample(clean, timesteps, noise)
    alpha_bar = diffusion.alpha_bars[timesteps].view(2, 1, 1, 1)
    expected = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    torch.testing.assert_close(observed, expected)


def test_shipped_schedules_reach_the_gaussian_prior() -> None:
    """Every notebook profile must train at a near-noise terminal timestep."""
    example_dir = Path(__file__).parents[3] / "examples" / "extraction" / "cifar10"
    train_config = yaml.safe_load((example_dir / "train_config.yaml").read_text(encoding="utf-8"))
    for name, values in train_config["profiles"].items():
        profile = RunProfile(name=name, seed=train_config["run"]["random_seed"], **values)
        diffusion = GaussianDiffusion(profile.timesteps, profile.sampling_steps, torch.device("cpu"))
        assert float(diffusion.alpha_bars[-1].sqrt()) < 0.01


def test_ddim_sampling_is_deterministic_and_invokes_each_guidance_step() -> None:
    """DDIM sampling must be seeded and apply guidance at every selected step."""
    torch.manual_seed(3)
    model = SmallTimeUNet(channels=8)
    diffusion = GaussianDiffusion(timesteps=6, sampling_steps=3, device=torch.device("cpu"))
    adapter = make_adapter(model, diffusion, torch.device("cpu"))

    first = adapter.sample(2, conditions=None, seed=11)
    second = adapter.sample(2, conditions=None, seed=11)
    guidance_calls = 0

    def gradient(images: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        nonlocal guidance_calls
        guidance_calls += 1
        assert timesteps.shape == labels.shape == (2,)
        return torch.full_like(images, 0.05)

    guided = adapter.sample_with_classifier_guidance(
        2,
        labels=torch.zeros(2, dtype=torch.long),
        gradient_fn=gradient,
        seed=11,
    )

    torch.testing.assert_close(first, second)
    assert guidance_calls == diffusion.sampling_steps
    assert not torch.equal(first, guided)


def test_train_then_reload_preserves_the_target_checkpoint(tmp_path: Path) -> None:
    """A matching checkpoint must reproduce the trained sampling weights."""
    profile = _test_profile()
    generator = torch.Generator(device="cpu").manual_seed(profile.seed)
    images = torch.rand((8, 3, 32, 32), generator=generator).mul(2.0).sub(1.0)
    dataset = TensorDataset(images, torch.zeros(8, dtype=torch.long))

    trained, _diffusion, checkpoint_path, losses = train_or_load_target(
        profile,
        dataset,
        tmp_path,
        torch.device("cpu"),
    )
    loaded, _diffusion, loaded_path, loaded_losses = train_or_load_target(
        profile,
        dataset,
        tmp_path,
        torch.device("cpu"),
    )

    assert checkpoint_path == loaded_path
    assert len(losses) == len(loaded_losses) == 1
    for trained_parameter, loaded_parameter in zip(trained.parameters(), loaded.parameters()):
        torch.testing.assert_close(trained_parameter, loaded_parameter)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["metadata"]["format_version"] -= 1
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="metadata does not match"):
        train_or_load_target(profile, dataset, tmp_path, torch.device("cpu"))


def test_composite_identity_tracks_each_output_determining_component() -> None:
    """Provider, feature, checkpoint, and reference changes must alter identity."""
    feature_module = torch.nn.Linear(3, 2)
    components = {
        "target_checkpoint_sha256": "checkpoint-a",
        "model_source_sha256": "model-a",
        "handler_source_sha256": "handler-a",
        "side_feature_state_sha256": sha256_module_state(feature_module),
        "side_feature_transform": "transform-a",
        "authorized_references_sha256": sha256_tensor(torch.zeros(2, 3, 4, 4)),
    }
    baseline = sha256_mapping(components)
    for name in components:
        changed = dict(components)
        changed[name] += "-changed"
        assert sha256_mapping(changed) != baseline

    notebook_path = Path(__file__).parents[3] / "examples" / "extraction" / "cifar10" / "main.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "'identity_components': identity_components" in notebook_source
    assert "'target_fingerprint': target_fingerprint" in notebook_source


def test_example_uses_leakpro_config_and_handler_layout(tmp_path: Path) -> None:
    """The real-data notebook must use checked-in configs and a separate handler."""
    example_dir = Path(__file__).parents[3] / "examples" / "extraction" / "cifar10"
    required_files = {
        ".gitignore",
        "audit.yaml",
        "audit_demonstration.yaml",
        "cifar10_handler.py",
        "cifar10_model.py",
        "main.ipynb",
        "train_config.yaml",
    }
    assert required_files.issubset(path.name for path in example_dir.iterdir())

    audit_config = yaml.safe_load((example_dir / "audit.yaml").read_text(encoding="utf-8"))
    assert audit_config["audit"]["attack_type"] == "extraction"
    assert [entry["attack"] for entry in audit_config["audit"]["attack_list"]] == [
        "carlini_diffusion",
        "side",
    ]
    train_config = yaml.safe_load((example_dir / "train_config.yaml").read_text(encoding="utf-8"))
    target_profile_keys = {
        "train_size",
        "epochs",
        "train_batch_size",
        "learning_rate",
        "timesteps",
        "sampling_steps",
        "model_channels",
        "reference_size",
    }
    assert all(set(profile) == target_profile_keys for profile in train_config["profiles"].values())
    assert set(train_config["run"]["audit_configs"]) == set(train_config["profiles"])

    changed_config = yaml.safe_load((example_dir / "audit.yaml").read_text(encoding="utf-8"))
    changed_config["audit"]["attack_list"][0]["num_unconditional_generations"] = 777
    changed_config["audit"]["attack_list"][1]["guidance_scale"] = 3.5
    changed_path = tmp_path / "audit.yaml"
    changed_path.write_text(yaml.safe_dump(changed_config), encoding="utf-8")
    resolved_config = load_audit_config(changed_path, target_fingerprint="sha256:test")
    assert resolved_config["audit"]["attack_list"][0]["num_unconditional_generations"] == 777
    assert resolved_config["audit"]["attack_list"][1]["guidance_scale"] == 3.5
    assert resolved_config["target"]["fingerprint"] == "sha256:test"

    notebook = json.loads((example_dir / "main.ipynb").read_text(encoding="utf-8"))
    notebook_source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "from cifar10_handler import CIFAR10ExtractionHandler, load_audit_config" in notebook_source
    assert "audit_config = load_audit_config(" in notebook_source
    assert "carlini_config['num_unconditional_generations'] =" not in notebook_source
    assert "side_config['num_generations'] =" not in notebook_source
    assert "Path('train_config.yaml')" in notebook_source


@pytest.mark.parametrize("config_name", ["audit.yaml", "audit_demonstration.yaml"])
def test_attack_random_seeds_are_explicit(config_name: str) -> None:
    """The root seed must also reach each independently validated attack config."""
    example_dir = Path(__file__).parents[3] / "examples" / "extraction" / "cifar10"
    audit_config = yaml.safe_load((example_dir / config_name).read_text(encoding="utf-8"))
    root_seed = audit_config["audit"]["random_seed"]
    attack_entries = {entry["attack"]: entry for entry in audit_config["audit"]["attack_list"]}
    carlini_values = {key: value for key, value in attack_entries["carlini_diffusion"].items() if key != "attack"}
    side_values = {key: value for key, value in attack_entries["side"].items() if key != "attack"}

    assert CarliniConfig(**carlini_values).random_seed == root_seed
    assert SIDEConfig(**side_values).random_seed == root_seed
