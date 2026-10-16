"""Model-stack boundary tests for callable and OpenAI-style adapters."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import nn

from leakpro.attacks.extraction_attacks import utils_generative
from leakpro.attacks.extraction_attacks.adapters import DiffusionAdapter
from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction
from leakpro.attacks.extraction_attacks.protocols import ExtractionAdapter
from leakpro.attacks.extraction_attacks.side import AttackSIDEExtraction


def test_hpu_seed_context_restores_device_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the Habana RNG API when an HPU sampler requests a seed."""
    class FakeHPURandom:
        state = [torch.tensor([5], dtype=torch.uint8)]

        def get_rng_state_all(self) -> list[torch.Tensor]:
            return [value.clone() for value in self.state]

        def manual_seed_all(self, seed: int) -> None:
            self.state = [torch.tensor([seed], dtype=torch.uint8)]

        def set_rng_state_all(self, states: list[torch.Tensor]) -> None:
            self.state = states

    fake_hpu = FakeHPURandom()
    monkeypatch.setattr(utils_generative, "import_module", lambda _: fake_hpu)
    cpu_state = torch.random.get_rng_state()

    with utils_generative.seeded_torch_rng(torch.device("hpu"), 7):
        assert fake_hpu.state[0].item() == 7
        torch.rand(1)

    assert fake_hpu.state[0].item() == 5
    assert torch.equal(torch.random.get_rng_state(), cpu_state)


def test_mps_seed_context_restores_device_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve MPS random state around a seeded sampling call."""
    mps_state = [torch.tensor([5], dtype=torch.uint8)]

    def seed_mps(seed: int) -> None:
        mps_state[0] = torch.tensor([seed], dtype=torch.uint8)

    def restore_mps(state: torch.Tensor, _device: torch.device) -> None:
        mps_state[0] = state

    monkeypatch.setattr(torch.mps, "get_rng_state", lambda _device: mps_state[0].clone())
    monkeypatch.setattr(torch.mps, "manual_seed", seed_mps)
    monkeypatch.setattr(torch.mps, "set_rng_state", restore_mps)
    cpu_state = torch.random.get_rng_state()

    with utils_generative.seeded_torch_rng(torch.device("mps"), 7):
        assert mps_state[0].item() == 7
        torch.rand(1)

    assert mps_state[0].item() == 5
    assert torch.equal(torch.random.get_rng_state(), cpu_state)


class DummyOpenAIDiffusion:
    """Implement the small improved-diffusion surface used by the adapter."""

    num_timesteps = 10

    @staticmethod
    def _scale_timesteps(timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps * 2

    @staticmethod
    def q_sample(clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        del timesteps
        return clean + noise

    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        *,
        clip_denoised: bool,
        model_kwargs: dict[str, object],
        device: torch.device,
        progress: bool,
        cond_fn: object | None = None,
    ) -> torch.Tensor:
        del model, clip_denoised, progress
        images = torch.zeros(shape, device=device)
        if cond_fn is not None:
            assert callable(cond_fn)
            raw_timesteps = torch.ones(shape[0], dtype=torch.long, device=device)
            return cond_fn(images, self._scale_timesteps(raw_timesteps), **model_kwargs)
        return images + float(model_kwargs.get("offset", 0.0))


class DummySpacedOpenAIDiffusion(DummyOpenAIDiffusion):
    """Faithfully expose reduced-to-original timestep mapping to cond_fn."""

    num_timesteps = 5
    timestep_map = [0, 2, 4, 6, 8]
    original_num_steps = 10
    rescale_timesteps = True

    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        *,
        clip_denoised: bool,
        model_kwargs: dict[str, object],
        device: torch.device,
        progress: bool,
        cond_fn: object | None = None,
    ) -> torch.Tensor:
        del model, clip_denoised, model_kwargs, progress
        images = torch.zeros(shape, device=device)
        if cond_fn is None:
            return images
        assert callable(cond_fn)
        reduced = torch.ones(shape[0], dtype=torch.long, device=device)
        mapped = torch.as_tensor(self.timestep_map, device=device)[reduced]
        scaled = mapped.float() * (1_000.0 / self.original_num_steps)
        return cond_fn(images, scaled)


class DummyImprovedOnlyDiffusion(DummyOpenAIDiffusion):
    """Expose Improved Diffusion's unguided p_sample_loop signature."""

    def __init__(self) -> None:
        self.sample_calls = 0

    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        *,
        clip_denoised: bool,
        model_kwargs: dict[str, object],
        device: torch.device,
        progress: bool,
    ) -> torch.Tensor:
        del model, clip_denoised, model_kwargs, progress
        self.sample_calls += 1
        return torch.zeros(shape, device=device)


def test_openai_adapter_sampling_forward_process_and_guidance() -> None:
    adapter = DiffusionAdapter(
        model=nn.Identity(),
        diffusion=DummyOpenAIDiffusion(),
        image_shape=(1, 2, 2),
        condition_encoder=lambda conditions: {"offset": float(len(conditions))},
    )
    sampled = adapter.sample(2, conditions=["a", "b"], seed=7)
    assert torch.equal(sampled, torch.full((2, 1, 2, 2), 2.0))
    torch.testing.assert_close(
        adapter.q_sample(torch.ones_like(sampled), torch.zeros(2, dtype=torch.long), torch.ones_like(sampled)),
        torch.full_like(sampled, 2.0),
    )
    assert adapter.classifier_timesteps(torch.tensor([1, 3])).tolist() == [2, 6]

    def gradient(images: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert timesteps.tolist() == [2, 2]
        return images + labels.float().view(-1, 1, 1, 1)

    guided = adapter.sample_with_classifier_guidance(
        2,
        labels=torch.tensor([1, 2]),
        gradient_fn=gradient,
        seed=8,
    )
    assert guided[:, 0, 0, 0].tolist() == [1.0, 2.0]


def test_openai_spaced_diffusion_training_and_guidance_timesteps_match() -> None:
    adapter = DiffusionAdapter(
        model=nn.Identity(),
        diffusion=DummySpacedOpenAIDiffusion(),
        image_shape=(1, 2, 2),
    )
    training_timesteps = adapter.classifier_timesteps(torch.tensor([1, 3]))
    observed: list[torch.Tensor] = []

    def gradient(images: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del labels
        observed.append(timesteps)
        return images

    adapter.sample_with_classifier_guidance(
        2,
        labels=torch.tensor([0, 1]),
        gradient_fn=gradient,
        seed=9,
    )

    assert training_timesteps.tolist() == [200.0, 600.0]
    assert observed[0].tolist() == [200.0, 200.0]


def test_side_rejects_improved_diffusion_loop_before_sampling() -> None:
    diffusion = DummyImprovedOnlyDiffusion()
    adapter = DiffusionAdapter(
        model=nn.Identity(),
        diffusion=diffusion,
        image_shape=(1, 2, 2),
    )
    attack = AttackSIDEExtraction(
        adapter,
        nn.Flatten(start_dim=1),
        {
            "synthetic_samples": 4,
            "clusters": 2,
            "min_cluster_size": 1,
        },
        target_name="improved-only-test",
    )

    with pytest.raises(ValueError, match="cond_fn"):
        attack.prepare_attack()

    assert diffusion.sample_calls == 0


def test_attacks_check_their_own_adapter_requirements() -> None:
    """Carlini accepts a sampler that lacks SIDE's white-box operations."""

    class ImageSampler(ExtractionAdapter[torch.Tensor]):
        model = nn.Identity()
        image_shape = (1, 4, 4)
        num_timesteps = 10

        def sample(self, batch_size: int, *, conditions: Sequence[object] | None, seed: int) -> torch.Tensor:
            del conditions, seed
            return torch.zeros((batch_size, *self.image_shape))

    sampler = ImageSampler()
    carlini = AttackCarliniExtraction(
        sampler,
        {"num_generations_per_condition": 2, "min_clique_size": 2},
        target_name="sampler-only",
        conditions=["label"],
    )
    carlini.prepare_attack()

    side = AttackSIDEExtraction(sampler, nn.Identity(), {}, target_name="sampler-only")
    with pytest.raises(TypeError, match="classifier_timesteps"):
        side.prepare_attack()


def test_vendored_diffusion_preserves_conditional_classifier_free_guidance() -> None:
    """The unconditional branch must not change MINV's labeled prediction."""
    from leakpro.attacks.utils.diff_mi.script_util import create_gaussian_diffusion

    class LabeledDenoiser(nn.Module):
        num_classes = 4

        def __init__(self) -> None:
            super().__init__()
            self.labels: list[torch.Tensor] = []

        def forward(self, images: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            self.labels.append(y.clone())
            return images * 0.05 + 0.1 + y[:, None, None, None] * 0.02

    model = LabeledDenoiser()
    diffusion = create_gaussian_diffusion(diffusion_steps=8, noise_schedule="cosine", w=0.5)
    images = torch.full((2, 1, 4, 4), 0.2)
    timesteps = torch.tensor([2, 5])
    labels = torch.tensor([0, 2])
    result = diffusion.p_mean_variance(model, images, timesteps, model_kwargs={"y": labels})
    conditional_eps = images * 0.05 + 0.1 + labels[:, None, None, None] * 0.02
    unconditional_eps = images * 0.05 + 0.1 + 3 * 0.02
    alpha = torch.tensor(diffusion.alphas_cumprod[timesteps], dtype=torch.float32).view(2, 1, 1, 1)
    expected_clean = ((images - (1 - alpha).sqrt() * (1.5 * conditional_eps - 0.5 * unconditional_eps)) / alpha.sqrt())

    torch.testing.assert_close(result["pred_xstart"], expected_clean.clamp(-1, 1))
    assert len(model.labels) == 2
    assert torch.equal(model.labels[0], labels)
    assert torch.equal(model.labels[1], torch.full_like(labels, 3))


def test_ddim_respacing_rejects_external_diffusion_before_sampling() -> None:
    """External enum/model contracts remain on their own original sampling path."""
    diffusion = DummyOpenAIDiffusion()
    with pytest.raises(TypeError, match="vendored GaussianDiffusion"):
        DiffusionAdapter(
            model=nn.Identity(), diffusion=diffusion, image_shape=(1, 4, 4), sampling_steps=3,
        )
    adapter = DiffusionAdapter(model=nn.Identity(), diffusion=diffusion, image_shape=(1, 4, 4))
    assert torch.equal(adapter.sample(2, conditions=None, seed=17), torch.zeros((2, 1, 4, 4)))
