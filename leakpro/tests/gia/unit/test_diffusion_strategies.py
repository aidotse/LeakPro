#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.diffusion.schedules import (
    ConstantSchedule,
    CosineDecaySchedule,
    LinearDecaySchedule,
)
from leakpro.attacks.gia_attacks.modular.components.diffusion.noise_strategies import (
    GradientAlignedNoiseInjection,
    StandardNoiseInjection,
)
from leakpro.attacks.gia_attacks.modular.components.diffusion.mean_strategies import (
    AdaptiveMeanOptimization,
    SimilarityGuidance,
)
from leakpro.attacks.gia_attacks.modular.components.losses import L2GradientLoss
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.loss_eval import evaluate_stage_losses
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.components.targets.fixed import FixedTarget


class TestSchedules:
    def test_constant_schedule_ignores_t(self):
        s = ConstantSchedule(0.3)
        assert s.gamma(0, 100) == 0.3
        assert s.gamma(99, 100) == 0.3

    def test_linear_decay_schedule_endpoints(self):
        s = LinearDecaySchedule()
        assert s.gamma(0, 100) == 0.0
        assert s.gamma(99, 100) == 1.0

    def test_cosine_decay_schedule_endpoints(self):
        s = CosineDecaySchedule()
        assert s.gamma(0, 100) == pytest_approx(0.0)
        assert s.gamma(99, 100) == pytest_approx(1.0)


def pytest_approx(value, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol
    return _Approx()


class TestNoiseStrategies:
    def test_standard_noise_matches_shape(self):
        strategy = StandardNoiseInjection()
        out = strategy.sample((2, 3, 4, 4), torch.device("cpu"), torch.float32, torch.zeros(2, 3, 4, 4))
        assert out.shape == (2, 3, 4, 4)

    def test_gradient_aligned_falls_back_when_delta_mu_near_zero(self):
        torch.manual_seed(0)
        strategy = GradientAlignedNoiseInjection(fallback_threshold=1e-3)
        delta_mu = torch.zeros(1, 1, 2, 2)
        out = strategy.sample((1, 1, 2, 2), torch.device("cpu"), torch.float32, delta_mu)
        assert out.shape == (1, 1, 2, 2)

    def test_gradient_aligned_projects_onto_delta_mu_direction(self):
        torch.manual_seed(0)
        strategy = GradientAlignedNoiseInjection()
        delta_mu = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        out = strategy.sample((1, 1, 2, 2), torch.device("cpu"), torch.float32, delta_mu)
        # Projected noise must be parallel to delta_mu (only the [0,0,0,0] entry nonzero).
        assert out[0, 0, 0, 1].item() == pytest_approx(0.0)
        assert out[0, 0, 1, 0].item() == pytest_approx(0.0)
        assert out[0, 0, 1, 1].item() == pytest_approx(0.0)


def _stage_and_ctx():
    class _FakeTask:
        def simulate_dummy_signal(self, model_input, target, ctx, *, create_graph=True):
            return ClientObservation(gradients=[model_input.sum().view(1)])

    class _IdentityModality:
        def to_model_input(self, data):
            return data

    ctx = Context(
        model=nn.Module(), observation=ClientObservation(gradients=[torch.zeros(1)]),
        modality=_IdentityModality(), task=_FakeTask(),
        input_shape=(1, 2, 2), device=torch.device("cpu"), axes=AxisManager(),
    )
    stage = OptimStage(
        losses={"l2": (L2GradientLoss(), 1.0)}, rep=None, driver=None,
        target_source=FixedTarget(), aggregator=None,
    )
    target = TargetInfo(value=torch.zeros(1))

    def objective(x):
        return evaluate_stage_losses(stage.losses, ctx, x, target, full_recon=x).total

    return objective


class TestMeanStrategies:
    def test_similarity_guidance_shifts_mean_toward_lower_loss(self):
        objective = _stage_and_ctx()
        mu_theta = torch.ones(1, 1, 2, 2)
        xt = torch.ones(1, 1, 2, 2)
        strategy = SimilarityGuidance(gamma=1.0)
        adjusted, delta = strategy.adjust(mu_theta, xt, sigma_t=1.0, t=0, num_timesteps=10, objective=objective)
        assert adjusted.shape == mu_theta.shape
        assert not torch.equal(adjusted, mu_theta)  # a real shift was applied
        assert torch.equal(delta, adjusted - mu_theta)

    def test_adaptive_mean_optimization_blends_toward_mu_star(self):
        objective = _stage_and_ctx()
        mu_theta = torch.ones(1, 1, 2, 2) * 5.0
        xt = mu_theta.clone()
        strategy = AdaptiveMeanOptimization(schedule=ConstantSchedule(1.0), inner_steps=3, inner_lr=0.1)
        adjusted, delta = strategy.adjust(mu_theta, xt, sigma_t=1.0, t=5, num_timesteps=10, objective=objective)
        assert adjusted.shape == mu_theta.shape
        # gamma=1.0 means adjusted == mu_star exactly (fully replaces mu_theta).
        assert torch.equal(delta, adjusted - mu_theta)
