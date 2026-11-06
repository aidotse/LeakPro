#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch
from torch import nn

diffusers = pytest.importorskip("diffusers")

from leakpro.attacks.gia_attacks.modular.components.diffusion.mean_strategies import SimilarityGuidance
from leakpro.attacks.gia_attacks.modular.components.diffusion.noise_strategies import StandardNoiseInjection
from leakpro.attacks.gia_attacks.modular.components.drivers.diffusion import DiffusionSamplingDriver
from leakpro.attacks.gia_attacks.modular.components.losses import L2GradientLoss
from leakpro.attacks.gia_attacks.modular.components.representations.diffusion import DiffusionRepresentation
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets.fixed import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, State, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.tests.gia.stand_ins import ZeroUNet, tiny_ddpm_scheduler


class _FakeTask:
    def simulate_dummy_signal(self, model_input, target, ctx, *, create_graph=True):
        return ClientObservation(gradients=[model_input.sum().view(1)])


class _IdentityModality:
    def to_model_input(self, data):
        return data


class TestDiffusionSamplingDriver:
    def test_run_returns_correct_shape_ddpm(self):
        device = torch.device("cpu")
        rep = DiffusionRepresentation()
        ctx = Context(
            model=nn.Module(), observation=ClientObservation(gradients=[torch.zeros(1)]),
            modality=_IdentityModality(), task=_FakeTask(),
            input_shape=(1, 1, 4, 4), device=device, axes=AxisManager(),
        )
        rep = rep.bind(State(), ctx)
        target_source = FixedTarget()
        target_source = target_source.bind(State(target=TargetInfo(value=torch.zeros(1))), ctx)
        stage = OptimStage(
            losses={"l2": (L2GradientLoss(), 1.0)}, rep=rep,
            driver=DiffusionSamplingDriver(
                unet=ZeroUNet(), scheduler=tiny_ddpm_scheduler(),
                mean_strategy=SimilarityGuidance(gamma=0.0), noise_strategy=StandardNoiseInjection(),
                sampler="ddpm",
            ),
            target_source=target_source, aggregator=None,
        )
        driver = stage.driver
        out = driver.run(stage, ctx)
        assert out.input.shape == (1, 1, 4, 4)

    def test_run_returns_correct_shape_ddim(self):
        device = torch.device("cpu")
        rep = DiffusionRepresentation()
        ctx = Context(
            model=nn.Module(), observation=ClientObservation(gradients=[torch.zeros(1)]),
            modality=_IdentityModality(), task=_FakeTask(),
            input_shape=(1, 1, 4, 4), device=device, axes=AxisManager(),
        )
        rep = rep.bind(State(), ctx)
        target_source = FixedTarget()
        target_source = target_source.bind(State(target=TargetInfo(value=torch.zeros(1))), ctx)
        stage = OptimStage(
            losses={"l2": (L2GradientLoss(), 1.0)}, rep=rep,
            driver=DiffusionSamplingDriver(
                unet=ZeroUNet(), scheduler=tiny_ddpm_scheduler(),
                mean_strategy=SimilarityGuidance(gamma=0.0), noise_strategy=StandardNoiseInjection(),
                sampler="ddim", num_steps=2,
            ),
            target_source=target_source, aggregator=None,
        )
        driver = stage.driver
        out = driver.run(stage, ctx)
        assert out.input.shape == (1, 1, 4, 4)


class TestDiffusionRepresentation:
    def _ctx(self):
        return Context(
            model=nn.Module(), observation=ClientObservation(gradients=[torch.zeros(1)]),
            modality=_IdentityModality(), task=_FakeTask(),
            input_shape=(1, 1, 4, 4), device=torch.device("cpu"), axes=AxisManager(),
        )

    def test_decode_maps_denoiser_space_to_data_space(self):
        rep = DiffusionRepresentation().bind(State(artifacts={"x_t": torch.full((1, 1, 4, 4), -0.5)}), self._ctx())
        torch.testing.assert_close(rep.decode(), torch.full((1, 1, 4, 4), 0.25))
        assert torch.equal(rep.artifacts()["x_t"], torch.full((1, 1, 4, 4), -0.5))

    def test_observable_clamps_x_t_to_the_denoiser_range(self):
        x_t = torch.tensor([-3.0, -1.0, 0.0, 2.0]).view(1, 1, 2, 2)
        rep = DiffusionRepresentation().bind(State(artifacts={"x_t": x_t}), self._ctx())
        torch.testing.assert_close(rep.to_observable(), torch.tensor([0.0, 0.0, 0.5, 1.0]).view(1, 1, 2, 2))

    def test_driver_rejects_other_representations(self):
        ctx = self._ctx()
        rep = PixelRepresentation().bind(State(), ctx)
        target_source = FixedTarget().bind(State(target=TargetInfo(value=torch.zeros(1))), ctx)
        driver = DiffusionSamplingDriver(
            unet=ZeroUNet(), scheduler=tiny_ddpm_scheduler(),
            mean_strategy=SimilarityGuidance(gamma=0.0), noise_strategy=StandardNoiseInjection(), sampler="ddim",
        )
        stage = OptimStage(losses={}, rep=rep, driver=driver, target_source=target_source, aggregator=None)
        with pytest.raises(TypeError, match="DiffusionRepresentation"):
            driver.run(stage, ctx)
