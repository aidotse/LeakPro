"""Checks for the runnable CIFAR-10 extraction notebook helper."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

# This optional example uses upstream sources that require an editable install.
pytest.importorskip("improved_diffusion", reason="Install examples/extraction/cifar10/requirements.txt")

from improved_diffusion.respace import SpacedDiffusion
from improved_diffusion.script_util import create_gaussian_diffusion
from improved_diffusion.unet import UNetModel
from torch.utils.data import TensorDataset

from examples.extraction.cifar10 import cifar10_model
from examples.extraction.cifar10.cifar10_handler import load_audit_config
from examples.extraction.cifar10.cifar10_model import (
    GaussianDiffusion,
    TrainConfig,
    make_adapter,
    sha256_module_state,
    sha256_tensor,
    train_or_load_target,
)
from leakpro.attacks.extraction_attacks.configs import CarliniConfig, SIDEConfig
from leakpro.utils.save_load import hash_config


def test_mac_auto_device_uses_cpu_for_the_official_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream float64 coefficient transfers are unsupported by MPS."""
    monkeypatch.setattr(cifar10_model, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert cifar10_model.select_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="supports CPU and CUDA"):
        cifar10_model.select_device("mps")
    with pytest.raises(ValueError, match="supports CPU and CUDA"):
        GaussianDiffusion(4, 4, torch.device("mps"))
    monkeypatch.setattr(cifar10_model, "get_device", lambda: torch.device("cuda"))
    assert cifar10_model.select_device("auto") == torch.device("cuda")


def _test_train_config() -> TrainConfig:
    return TrainConfig(
        seed=7,
        train_size=8,
        epochs=1,
        train_batch_size=4,
        learning_rate=1e-3,
        timesteps=4,
        sampling_steps=4,
        model_channels=32,
        num_res_blocks=1,
        dropout=0.0,
        microbatch=2,
        ema_decay=0.9,
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
    """The configured schedule must reach a near-noise terminal timestep."""
    example_dir = Path(__file__).parents[3] / "examples" / "extraction" / "cifar10"
    train_config = yaml.safe_load((example_dir / "train_config.yaml").read_text(encoding="utf-8"))
    train = TrainConfig(seed=train_config["run"]["random_seed"], **train_config["train"])
    diffusion = GaussianDiffusion(train.timesteps, train.sampling_steps, torch.device("cpu"))
    assert float(diffusion.alpha_bars[-1].sqrt()) < 0.01


class _TinyLearnedVarianceModel(torch.nn.Module):
    """Keep sampler and accumulation checks independent of U-Net capacity."""

    def __init__(self) -> None:
        super().__init__()
        self.output_scale = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.3, 0.1, 0.2, -0.2]))
        self.seen_timesteps: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        self.seen_timesteps.append(timesteps.detach().clone())
        return torch.cat((images, torch.ones_like(images)), dim=1) * self.output_scale[None, :, None, None]


def _official_diffusion(steps: int, sampling_steps: int | None = None) -> SpacedDiffusion:
    return create_gaussian_diffusion(
        steps=steps,
        learn_sigma=True,
        noise_schedule="cosine",
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        timestep_respacing="" if sampling_steps is None else [sampling_steps],
    )


def test_ddim_sampling_matches_openai_and_uses_native_guidance_timesteps() -> None:
    """Respacing must preserve the published sampler and SIDE's noise levels."""
    model = _TinyLearnedVarianceModel()
    diffusion = GaussianDiffusion(timesteps=6, sampling_steps=3, device=torch.device("cpu"))
    adapter = make_adapter(model, diffusion, torch.device("cpu"))
    noise = torch.randn((2, 3, 32, 32), generator=torch.Generator().manual_seed(11))
    oracle = _official_diffusion(6, 3)
    with torch.no_grad():
        expected = oracle.ddim_sample_loop(model, noise.shape, noise=noise, device=torch.device("cpu"))
    model.seen_timesteps.clear()

    first = adapter.sample(2, conditions=None, seed=11)
    torch.testing.assert_close(first, expected)
    expected_native_steps = list(reversed(oracle.timestep_map))
    assert len(model.seen_timesteps) == len(expected_native_steps)
    for observed, native in zip(model.seen_timesteps, expected_native_steps):
        torch.testing.assert_close(observed, torch.full((2,), native * 1000.0 / 6))
    second = adapter.sample(2, conditions=None, seed=11)
    guidance_steps: list[int] = []

    def gradient(images: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert timesteps.shape == labels.shape == (2,)
        assert timesteps.dtype == torch.long
        assert torch.equal(timesteps, timesteps[0].expand_as(timesteps))
        guidance_steps.append(int(timesteps[0]))
        return torch.full_like(images, 0.05)

    guided = adapter.sample_with_classifier_guidance(
        2,
        labels=torch.zeros(2, dtype=torch.long),
        gradient_fn=gradient,
        seed=11,
    )

    torch.testing.assert_close(first, second)
    assert guidance_steps == expected_native_steps
    assert not torch.equal(first, guided)


def _guided_reference_ddim(
    diffusion: GaussianDiffusion,
    model: torch.nn.Module,
    gradient_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    labels: torch.Tensor,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    """Reference guided loop written from the official OpenAI samplers.

    The eps shift matches guided_diffusion.gaussian_diffusion.condition_score.
    The recomputed x0 is re-clamped before the eta-zero update because the
    spaced cosine schedule multiplies any eps change by about 1.3e3 at the
    terminal step, which would otherwise push the state far outside the range
    the target was trained on.
    """
    sampler = diffusion.sampler
    images = torch.randn(
        (batch_size, 3, 32, 32), generator=torch.Generator(device="cpu").manual_seed(seed)
    )
    model.eval()
    for index in reversed(range(diffusion.sampling_steps)):
        timesteps = torch.full((batch_size,), index, dtype=torch.long)
        with torch.no_grad():
            output = sampler.p_mean_variance(model, images, timesteps, clip_denoised=True)
            predicted_noise = sampler._predict_eps_from_xstart(images, timesteps, output["pred_xstart"])
            gradient = gradient_fn(images, timesteps, labels)
            alpha = float(sampler.alphas_cumprod[index])
            predicted_noise = predicted_noise - (1.0 - alpha) ** 0.5 * gradient
            predicted_clean = sampler._predict_xstart_from_eps(images, timesteps, predicted_noise)
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)
            predicted_noise = sampler._predict_eps_from_xstart(images, timesteps, predicted_clean)
            previous_alpha = float(sampler.alphas_cumprod_prev[index])
            images = previous_alpha**0.5 * predicted_clean + (1.0 - previous_alpha) ** 0.5 * predicted_noise
    return images.clamp(-1.0, 1.0)


def _terminal_guidance_setup() -> tuple[_TinyLearnedVarianceModel, GaussianDiffusion, torch.Tensor]:
    """Use the shipped 4,000-step cosine schedule so the terminal step is exercised."""
    model = _TinyLearnedVarianceModel()
    diffusion = GaussianDiffusion(timesteps=4000, sampling_steps=100, device=torch.device("cpu"))
    return model, diffusion, torch.zeros(2, dtype=torch.long)


def test_guided_sampling_reclamps_the_guided_clean_prediction() -> None:
    """Unclamped guidance saturates every sample at the terminal cosine step."""
    model, diffusion, labels = _terminal_guidance_setup()

    def gradient(images: torch.Tensor, _timesteps: torch.Tensor, _labels: torch.Tensor) -> torch.Tensor:
        return torch.full_like(images, 0.2)

    guided = diffusion.sample(model, 2, seed=11, labels=labels, gradient_fn=gradient)
    reference = _guided_reference_ddim(diffusion, model, gradient, labels, 2, 11)

    torch.testing.assert_close(guided, reference)
    assert torch.isfinite(guided).all()
    assert float((guided.abs() > 0.99).float().mean()) < 0.8
    assert abs(float(guided.mean())) < 0.5


def test_zero_guidance_matches_unguided_sampling() -> None:
    """A zero classifier gradient must reproduce the official DDIM path."""
    model, diffusion, labels = _terminal_guidance_setup()
    unguided = diffusion.sample(model, 2, seed=13)
    zero_guided = diffusion.sample(
        model,
        2,
        seed=13,
        labels=labels,
        gradient_fn=lambda images, _timesteps, _labels: torch.zeros_like(images),
    )
    torch.testing.assert_close(zero_guided, unguided, atol=0.0, rtol=0.0)


def test_classifier_guidance_moves_samples_along_the_gradient() -> None:
    """Ascending classifier gradients must shift samples toward higher logits."""
    model, diffusion, labels = _terminal_guidance_setup()

    def make_gradient(sign: float) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        def gradient(images: torch.Tensor, _timesteps: torch.Tensor, _labels: torch.Tensor) -> torch.Tensor:
            return torch.full_like(images, sign)

        return gradient

    unguided = diffusion.sample(model, 2, seed=17)
    positive = diffusion.sample(model, 2, seed=17, labels=labels, gradient_fn=make_gradient(0.2))
    negative = diffusion.sample(model, 2, seed=17, labels=labels, gradient_fn=make_gradient(-0.2))

    assert float((positive - unguided).mean()) > 0.1
    assert float((negative - unguided).mean()) < -0.1


def test_training_loss_matches_the_published_hybrid_objective() -> None:
    """Learned variance must receive its variational loss as well as epsilon MSE."""
    model = _TinyLearnedVarianceModel()
    diffusion = GaussianDiffusion(4, 4, torch.device("cpu"))
    generator = torch.Generator().manual_seed(4)
    images = torch.rand((2, 3, 4, 4), generator=generator).mul(2).sub(1)
    noise = torch.randn(images.shape, generator=generator)
    timesteps = torch.tensor([0, 3])
    observed = diffusion.process.training_losses(model, images, timesteps, noise=noise)
    expected = _official_diffusion(4).training_losses(model, images, timesteps, noise=noise)

    for name in ("loss", "mse", "vb"):
        torch.testing.assert_close(observed[name], expected[name])
    torch.testing.assert_close(observed["loss"], observed["mse"] + observed["vb"])
    observed["loss"].mean().backward()
    assert model.output_scale.grad is not None
    assert torch.isfinite(model.output_scale.grad).all()
    assert torch.count_nonzero(model.output_scale.grad[3:]) == 3


def test_microbatches_preserve_the_effective_batch_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Uneven microbatches and a partial final batch must retain sample weights."""
    monkeypatch.setattr(
        cifar10_model, "create_target_model", lambda _train, device: _TinyLearnedVarianceModel().to(device)
    )
    train = replace(_test_train_config(), train_size=130, train_batch_size=128, microbatch=128)
    generator = torch.Generator().manual_seed(8)
    images = torch.rand((130, 3, 4, 4), generator=generator).mul(2).sub(1)
    dataset = TensorDataset(images, torch.zeros(130, dtype=torch.long))
    full, _, _, full_losses = train_or_load_target(train, dataset, tmp_path / "full", torch.device("cpu"))
    accumulated, _, _, accumulated_losses = train_or_load_target(
        replace(train, microbatch=17), dataset, tmp_path / "micro", torch.device("cpu")
    )

    torch.testing.assert_close(torch.tensor(full_losses), torch.tensor(accumulated_losses))
    torch.testing.assert_close(full.output_scale, accumulated.output_scale)


def test_train_then_reload_preserves_the_target_checkpoint(tmp_path: Path) -> None:
    """A matching checkpoint must reproduce the trained sampling weights."""
    train = _test_train_config()
    generator = torch.Generator(device="cpu").manual_seed(train.seed)
    images = torch.rand((8, 3, 32, 32), generator=generator).mul(2.0).sub(1.0)
    dataset = TensorDataset(images, torch.zeros(8, dtype=torch.long))

    trained, _diffusion, checkpoint_path, losses = train_or_load_target(
        train,
        dataset,
        tmp_path,
        torch.device("cpu"),
    )
    loaded, _diffusion, loaded_path, loaded_losses = train_or_load_target(
        train,
        dataset,
        tmp_path,
        torch.device("cpu"),
    )

    assert isinstance(trained, UNetModel)
    assert checkpoint_path == loaded_path
    assert torch.isfinite(torch.tensor(losses)).all()
    assert len(losses) == len(loaded_losses) == 1
    for trained_parameter, loaded_parameter in zip(trained.parameters(), loaded.parameters()):
        torch.testing.assert_close(trained_parameter, loaded_parameter)

    resampled, resampled_diffusion, resampled_path, _ = train_or_load_target(
        replace(train, sampling_steps=2), dataset, tmp_path, torch.device("cpu")
    )
    assert resampled_path == checkpoint_path
    assert resampled_diffusion.sampling_steps == 2
    assert sha256_module_state(resampled) == sha256_module_state(trained)
    with pytest.raises(ValueError, match="metadata does not match"):
        train_or_load_target(replace(train, learning_rate=2e-3), dataset, tmp_path, torch.device("cpu"))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["metadata"]["format_version"] -= 1
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="metadata does not match"):
        train_or_load_target(train, dataset, tmp_path, torch.device("cpu"))


def test_interrupted_training_resumes_optimizer_ema_and_random_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming a saved epoch must equal uninterrupted training with dropout."""

    class DropoutModel(_TinyLearnedVarianceModel):
        def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            images = torch.nn.functional.dropout(images, p=0.2, training=self.training)
            return super().forward(images, timesteps)

    monkeypatch.setattr(cifar10_model, "create_target_model", lambda _train, device: DropoutModel().to(device))
    monkeypatch.setattr(cifar10_model, "CHECKPOINT_EVERY_EPOCHS", 1)
    train = replace(_test_train_config(), train_size=5, reference_size=5, epochs=3, dropout=0.2)
    images = torch.rand((5, 3, 4, 4), generator=torch.Generator().manual_seed(9)).mul(2).sub(1)
    dataset = TensorDataset(images, torch.zeros(5, dtype=torch.long))
    continuous, _, _, continuous_losses = train_or_load_target(
        train, dataset, tmp_path / "continuous", torch.device("cpu")
    )
    save_checkpoint = cifar10_model._save_checkpoint

    def save_then_interrupt(path: Path, state: dict) -> None:
        save_checkpoint(path, state)
        if path.name.endswith(".resume.pt"):
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(cifar10_model, "_save_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_or_load_target(train, dataset, tmp_path / "interrupted", torch.device("cpu"))
    assert not (tmp_path / "interrupted" / "cifar10_ddpm.pt").exists()
    assert (tmp_path / "interrupted" / "cifar10_ddpm.resume.pt").exists()

    monkeypatch.setattr(cifar10_model, "_save_checkpoint", save_checkpoint)
    resumed, _, checkpoint_path, resumed_losses = train_or_load_target(
        train, dataset, tmp_path / "interrupted", torch.device("cpu")
    )
    assert resumed_losses == continuous_losses
    torch.testing.assert_close(resumed.output_scale, continuous.output_scale, rtol=0, atol=0)
    assert checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".resume.pt").exists()


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
    baseline = hash_config(components)
    for name in components:
        changed = dict(components)
        changed[name] += "-changed"
        assert hash_config(changed) != baseline

    notebook_path = Path(__file__).parents[3] / "examples" / "extraction" / "cifar10" / "main.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "'identity_components': identity_components" in notebook_source
    assert "'target_hash': target_hash" in notebook_source


def test_example_uses_leakpro_config_and_handler_layout(tmp_path: Path) -> None:
    """The real-data notebook must use checked-in configs and a separate handler."""
    example_dir = Path(__file__).parents[3] / "examples" / "extraction" / "cifar10"
    required_files = {
        ".gitignore",
        "audit.yaml",
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
    target_train_keys = {
        "train_size",
        "epochs",
        "train_batch_size",
        "learning_rate",
        "timesteps",
        "sampling_steps",
        "model_channels",
        "reference_size",
        "num_res_blocks",
        "dropout",
        "microbatch",
        "ema_decay",
    }
    assert set(train_config) == {"run", "train"}
    assert set(train_config["train"]) == target_train_keys
    assert (example_dir / train_config["run"]["audit_config"]).is_file()

    changed_config = yaml.safe_load((example_dir / "audit.yaml").read_text(encoding="utf-8"))
    changed_config["audit"]["attack_list"][0]["num_unconditional_generations"] = 777
    changed_config["audit"]["attack_list"][1]["guidance_scale"] = 3.5
    changed_path = tmp_path / "audit.yaml"
    changed_path.write_text(yaml.safe_dump(changed_config), encoding="utf-8")
    resolved_config = load_audit_config(changed_path, target_hash="sha256:test")
    assert resolved_config["audit"]["attack_list"][0]["num_unconditional_generations"] == 777
    assert resolved_config["audit"]["attack_list"][1]["guidance_scale"] == 3.5
    assert resolved_config["target"]["hash"] == "sha256:test"

    notebook = json.loads((example_dir / "main.ipynb").read_text(encoding="utf-8"))
    notebook_source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "from cifar10_handler import CIFAR10ExtractionHandler, load_audit_config" in notebook_source
    assert "audit_config = load_audit_config(" in notebook_source
    assert "carlini_config['num_unconditional_generations'] =" not in notebook_source
    assert "side_config['num_generations'] =" not in notebook_source
    assert "Path('train_config.yaml')" in notebook_source


def test_attack_random_seeds_are_explicit() -> None:
    """The root seed must also reach each independently validated attack config."""
    example_dir = Path(__file__).parents[3] / "examples" / "extraction" / "cifar10"
    audit_config = yaml.safe_load((example_dir / "audit.yaml").read_text(encoding="utf-8"))
    root_seed = audit_config["audit"]["random_seed"]
    attack_entries = {entry["attack"]: entry for entry in audit_config["audit"]["attack_list"]}
    config_types = {"carlini_diffusion": CarliniConfig, "side": SIDEConfig}
    for name, entry in attack_entries.items():
        values = {key: value for key, value in entry.items() if key != "attack"}
        assert config_types[name](**values).random_seed == root_seed


def test_example_uses_shared_seeding(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared helper controls backend determinism as well as RNG seeds."""
    from leakpro.utils.seed import seed_everything

    assert cifar10_model.seed_everything is seed_everything
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    cifar10_model.seed_everything(7)
    first = torch.rand(3)
    cifar10_model.seed_everything(7)
    torch.testing.assert_close(first, torch.rand(3))
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
