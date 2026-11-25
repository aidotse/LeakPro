#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""DDPM posterior for a reverse step t -> t_prev that may skip timesteps (subsampled sampling)."""
import pytest
import torch
from torch import nn

diffusers = pytest.importorskip("diffusers")

from leakpro.attacks.gia_attacks.modular.components.diffusion.mean_strategies import SimilarityGuidance
from leakpro.attacks.gia_attacks.modular.components.diffusion.noise_strategies import StandardNoiseInjection
from leakpro.attacks.gia_attacks.modular.components.drivers.diffusion import DiffusionSamplingDriver
from leakpro.attacks.gia_attacks.modular.components.losses import L2GradientLoss
from leakpro.attacks.gia_attacks.modular.components.representations.diffusion import DiffusionRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets.fixed import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, State, TargetInfo
from leakpro.fl_utils.diffusion_handler import DDPMSchedulerWrapper
from leakpro.tests.gia.stand_ins import ZeroUNet


def _wrapper(num_train_timesteps: int = 1000) -> DDPMSchedulerWrapper:
    return DDPMSchedulerWrapper(diffusers.DDPMScheduler(num_train_timesteps=num_train_timesteps), device=torch.device("cpu"))


def _brute_force_posterior(betas: torch.Tensor, t: int, t_prev: int) -> tuple[float, float, float]:
    """Coefficients of q(x_{t_prev} | x_t, x_0) for a scalar chain built from one-step Gaussians.

    Starting at x_0 (fixed), x_s = sqrt(1 - beta_s) x_{s-1} + sqrt(beta_s) eps_s. Track the joint law of
    (x_{t_prev}, x_t) as linear functions of x_0 plus Gaussian noise, then condition with the textbook formula.
    """
    mean_coef = 1.0  # E[x_s] = mean_coef * x_0
    var = 0.0
    for s in range(t_prev + 1):
        mean_coef, var = mean_coef * (1 - betas[s]) ** 0.5, var * (1 - betas[s]) + betas[s]
    m_prev, v_prev = mean_coef, var
    gain = 1.0  # x_t = gain * x_{t_prev} + independent noise
    for s in range(t_prev + 1, t + 1):
        gain, var = gain * (1 - betas[s]) ** 0.5, var * (1 - betas[s]) + betas[s]
    m_t, v_t = mean_coef * gain, var
    cov = gain * v_prev  # Cov(x_{t_prev}, x_t)
    # E[x_{t_prev} | x_t, x_0] = m_prev x_0 + cov / v_t (x_t - m_t x_0)
    c_xt = cov / v_t
    c_x0 = m_prev - c_xt * m_t
    return float(c_x0), float(c_xt), float(v_prev - cov**2 / v_t)


@pytest.mark.parametrize(("t", "t_prev"), [(990, 979), (500, 400), (37, 0), (1, 0), (999, 998)])
def test_jump_posterior_matches_composed_one_step_chain(t: int, t_prev: int) -> None:
    sched = _wrapper()
    c_x0, c_xt, var = _brute_force_posterior(sched.betas.tolist(), t, t_prev)
    x0, xt = torch.tensor([1.0]), torch.tensor([0.0])
    assert sched.posterior_mean(x0, xt, t, t_prev).item() == pytest.approx(c_x0, rel=1e-4, abs=1e-6)
    assert sched.posterior_mean(torch.tensor([0.0]), torch.tensor([1.0]), t, t_prev).item() == pytest.approx(c_xt, rel=1e-4)
    assert sched.posterior_std(t, t_prev).item() ** 2 == pytest.approx(var, rel=1e-3, abs=1e-9)


def test_one_step_reduces_to_the_standard_posterior() -> None:
    sched = _wrapper()
    x0, xt = torch.randn(2, 3), torch.randn(2, 3)
    for t in (1, 10, 500, 999):
        torch.testing.assert_close(sched.posterior_mean(x0, xt, t, t - 1), sched.posterior_mean(x0, xt, t))
        torch.testing.assert_close(sched.posterior_std(t, t - 1), sched.posterior_variance[t].float().sqrt())
        torch.testing.assert_close(
            sched.step_mean(x0, xt, torch.zeros_like(xt), t, t - 1, sampler="ddpm"), sched.posterior_mean(x0, xt, t)
        )


def test_jump_differs_from_the_one_step_posterior() -> None:
    sched = _wrapper()
    # 990 -> 979 skips ten timesteps: the posterior variance is about ten times the one-step value.
    assert sched.posterior_std(990, 979).item() ** 2 == pytest.approx(0.197, abs=0.01)
    assert sched.posterior_std(990).item() ** 2 == pytest.approx(0.0198, abs=0.001)
    x0, xt = torch.zeros(1), torch.ones(1)
    assert sched.step_mean(x0, xt, torch.zeros(1), 990, 979, sampler="ddpm").item() == pytest.approx(0.896, abs=0.005)


def test_final_step_to_the_clean_image_has_zero_variance() -> None:
    sched = _wrapper()
    assert sched.posterior_std(0, -1).item() == 0.0
    assert sched.posterior_std(5, -1).item() == 0.0
    x0, xt = torch.randn(4), torch.randn(4)
    torch.testing.assert_close(sched.posterior_mean(x0, xt, 5, -1), x0)


class _RecordingScheduler(DDPMSchedulerWrapper):
    def __init__(self) -> None:
        super().__init__(diffusers.DDPMScheduler(num_train_timesteps=20), device=torch.device("cpu"))
        self.std_calls: list[tuple[int, int | None]] = []
        self.mean_calls: list[tuple[int, int]] = []

    def posterior_std(self, t: int, t_prev: int | None = None) -> torch.Tensor:
        self.std_calls.append((t, t_prev))
        return super().posterior_std(t, t_prev)

    def step_mean(self, x0_pred, xt, eps_pred, t, t_prev, sampler="ddpm"):  # noqa: ANN001, ANN201
        self.mean_calls.append((t, t_prev))
        return super().step_mean(x0_pred, xt, eps_pred, t, t_prev, sampler=sampler)


class _FakeTask:
    def simulate_dummy_signal(self, model_input, target, ctx, *, create_graph=True):  # noqa: ANN001, ANN201, ARG002
        return ClientObservation(gradients=[model_input.sum().view(1)])


class _IdentityModality:
    def to_model_input(self, data):  # noqa: ANN001, ANN201
        return data


def test_subsampled_ddpm_driver_uses_the_jump_posterior() -> None:
    sched = _RecordingScheduler()
    ctx = Context(
        model=nn.Module(), observation=ClientObservation(gradients=[torch.zeros(1)]), modality=_IdentityModality(),
        task=_FakeTask(), input_shape=(1, 1, 4, 4), device=torch.device("cpu"), axes=AxisManager(),
    )
    rep = DiffusionRepresentation().bind(State(), ctx)
    target_source = FixedTarget().bind(State(target=TargetInfo(value=torch.zeros(1))), ctx)
    driver = DiffusionSamplingDriver(
        unet=ZeroUNet(), scheduler=sched, mean_strategy=SimilarityGuidance(gamma=0.0),
        noise_strategy=StandardNoiseInjection(), sampler="ddpm", num_steps=5,
    )
    stage = OptimStage(losses={"l2": (L2GradientLoss(), 1.0)}, rep=rep, driver=driver, target_source=target_source,
                       aggregator=None)
    driver.run(stage, ctx)
    waypoints = driver._build_waypoints(20)  # noqa: SLF001
    pairs = list(zip(waypoints, waypoints[1:] + [-1]))
    assert sched.std_calls == pairs
    assert sched.mean_calls == pairs[:-1]
    assert any(t - t_prev > 1 for t, t_prev in pairs[:-1])
