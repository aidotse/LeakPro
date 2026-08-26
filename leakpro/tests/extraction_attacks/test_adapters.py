"""Model-stack boundary tests for callable and OpenAI-style adapters."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import nn

from leakpro.attacks.extraction_attacks.adapters import CallableDiffusionAdapter, OpenAIDiffusionAdapter
from leakpro.attacks.extraction_attacks.side import AttackSIDEExtraction


def test_callable_adapter_scales_guidance_timesteps() -> None:
    observed: list[torch.Tensor] = []

    def sample(batch_size: int, conditions: Sequence[object] | None, seed: int) -> torch.Tensor:
        del conditions, seed
        return torch.zeros((batch_size, 1, 2, 2))

    def guided_sample(batch_size: int, labels: torch.Tensor, gradient_fn: object, seed: int) -> torch.Tensor:
        del seed
        assert callable(gradient_fn)
        images = torch.zeros((batch_size, 1, 2, 2))
        return gradient_fn(images, torch.tensor([2] * batch_size), labels)

    def gradient(images: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del labels
        observed.append(timesteps)
        return images

    adapter = CallableDiffusionAdapter(
        image_shape=(1, 2, 2),
        sample_fn=sample,
        num_timesteps=10,
        guided_sample_fn=guided_sample,
        classifier_timestep_fn=lambda timesteps: timesteps * 3,
    )
    output = adapter.sample_with_classifier_guidance(
        2,
        labels=torch.tensor([0, 1]),
        gradient_fn=gradient,
        seed=4,
    )
    assert output.shape == (2, 1, 2, 2)
    assert observed[0].tolist() == [6, 6]
    with pytest.raises(NotImplementedError, match="q_sample_fn"):
        adapter.q_sample(output, torch.zeros(2, dtype=torch.long), output)


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


class DummyPositionalOnlyGuidanceDiffusion(DummyOpenAIDiffusion):
    """Expose an incompatible positional-only cond_fn despite accepting other keywords."""

    def __init__(self) -> None:
        self.sample_calls = 0

    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        cond_fn: object,
        /,
        **kwargs: object,
    ) -> torch.Tensor:
        del model, cond_fn, kwargs
        self.sample_calls += 1
        return torch.zeros(shape)


def test_openai_adapter_sampling_forward_process_and_guidance() -> None:
    adapter = OpenAIDiffusionAdapter(
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


def test_cpu_openai_adapter_does_not_seed_accelerators(monkeypatch: pytest.MonkeyPatch) -> None:
    accelerator_seed_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: accelerator_seed_calls.append(("cuda", seed)),
    )
    monkeypatch.setattr(
        torch.mps,
        "manual_seed",
        lambda seed: accelerator_seed_calls.append(("mps", seed)),
    )
    adapter = OpenAIDiffusionAdapter(
        model=nn.Identity(),
        diffusion=DummyOpenAIDiffusion(),
        image_shape=(1, 2, 2),
    )
    cpu_state = torch.random.get_rng_state()

    adapter.sample(2, conditions=None, seed=73)

    assert accelerator_seed_calls == []
    assert torch.equal(torch.random.get_rng_state(), cpu_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for RNG isolation coverage.")
def test_cuda_openai_adapter_restores_every_cuda_rng_state() -> None:
    adapter = OpenAIDiffusionAdapter(
        model=nn.Identity(),
        diffusion=DummyOpenAIDiffusion(),
        image_shape=(1, 2, 2),
        device="cuda:0",
    )
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()

    adapter.sample(2, conditions=None, seed=79)

    assert torch.equal(torch.random.get_rng_state(), cpu_state)
    assert len(torch.cuda.get_rng_state_all()) == len(cuda_states)
    assert all(torch.equal(actual, expected) for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_states))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required for RNG isolation coverage.")
def test_mps_openai_adapter_restores_selected_rng_state() -> None:
    adapter = OpenAIDiffusionAdapter(
        model=nn.Identity(),
        diffusion=DummyOpenAIDiffusion(),
        image_shape=(1, 2, 2),
        device="mps",
    )
    cpu_state = torch.random.get_rng_state()
    mps_state = torch.mps.get_rng_state()

    adapter.sample(2, conditions=None, seed=83)

    assert torch.equal(torch.random.get_rng_state(), cpu_state)
    assert torch.equal(torch.mps.get_rng_state(), mps_state)


def test_openai_spaced_diffusion_training_and_guidance_timesteps_match() -> None:
    adapter = OpenAIDiffusionAdapter(
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


def test_openai_adapter_rejects_invalid_or_overlapping_model_kwargs() -> None:
    with pytest.raises(TypeError, match="q_sample"):
        OpenAIDiffusionAdapter(model=nn.Identity(), diffusion=object(), image_shape=(1, 2, 2))

    adapter = OpenAIDiffusionAdapter(
        model=nn.Identity(),
        diffusion=DummyOpenAIDiffusion(),
        image_shape=(1, 2, 2),
        condition_encoder=lambda conditions: {"offset": len(conditions)},
        base_model_kwargs={"offset": 1},
    )
    with pytest.raises(ValueError, match="overwrote"):
        adapter.sample(1, conditions=["a"], seed=3)


def test_side_rejects_improved_diffusion_loop_before_sampling() -> None:
    diffusion = DummyImprovedOnlyDiffusion()
    adapter = OpenAIDiffusionAdapter(
        model=nn.Identity(),
        diffusion=diffusion,
        image_shape=(1, 2, 2),
    )
    attack = AttackSIDEExtraction(
        adapter,
        nn.Flatten(start_dim=1),
        {
            "authorized_audit": True,
            "synthetic_samples": 4,
            "clusters": 2,
            "min_cluster_size": 1,
        },
        audit_fingerprint="improved-only-test",
    )

    with pytest.raises(ValueError, match="cond_fn"):
        attack.prepare_attack()

    assert diffusion.sample_calls == 0


def test_side_rejects_positional_only_cond_fn_before_sampling() -> None:
    diffusion = DummyPositionalOnlyGuidanceDiffusion()
    adapter = OpenAIDiffusionAdapter(
        model=nn.Identity(),
        diffusion=diffusion,
        image_shape=(1, 2, 2),
    )
    attack = AttackSIDEExtraction(
        adapter,
        nn.Flatten(start_dim=1),
        {
            "authorized_audit": True,
            "synthetic_samples": 4,
            "clusters": 2,
            "min_cluster_size": 1,
        },
        audit_fingerprint="positional-only-guidance-test",
    )

    with pytest.raises(ValueError, match="cond_fn"):
        attack.prepare_attack()

    assert diffusion.sample_calls == 0
