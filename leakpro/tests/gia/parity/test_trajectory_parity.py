#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The framework, configured v1-equivalent, reproduces v1's recorded iterates z_t and losses L_t.

z_t is the iterate at which L_t is evaluated. The driver reports L_t for step t (the loss at the
iterate the step started from) and the iterate *after* step t in the event artifacts, so the
iterate after step t is z_{t+1}.
"""
import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.events import StepEvent
from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback
from leakpro.tests.gia import helpers
from leakpro.tests.gia.parity import fixtures, v1_equivalent
from leakpro.tests.gia.parity.deltas import delta

EARLY = 10

# The fixtures were recorded on one machine. CPU kernels round the last float32 bit differently depending on the torch
# release, the CPU instruction set and the thread count (about 1e-6 relative on the gradients), so the comparison is
# always tolerance based. A 1e-6 relative perturbation of the gradients moves Adam iterates by about 3e-4 and L-BFGS
# iterates by O(1) after a few steps (L-BFGS amplifies round-off through its curvature history), so for L-BFGS only the
# first steps are comparable. A wrong formula changes these quantities by far more than the tolerances below.
ITERATE_TOL = {"rtol": 1e-3, "atol": 1e-4}
EARLY_LOSS_RTOL = 1e-4
LBFGS_COMPARABLE_STEPS = 2   # iterates z_0..z_2 and losses L_0..L_3


class _Trajectory(Callback):
    def __init__(self) -> None:
        self.losses: list[float] = []
        self.iterates: dict[int, torch.Tensor] = {}

    def on_step_end(self, event: StepEvent) -> None:
        self.losses.append(event.loss)
        self.iterates[event.step + 1] = event.artifacts["model_input"].detach().clone()


def _replay(case_name: str) -> tuple[dict, _Trajectory, v1_equivalent.TrajectoryCase]:
    fx = fixtures.load(case_name)
    case = v1_equivalent.trajectory_case(fx)
    recorder = _Trajectory()
    run_attack(case.attack, case.model, case.observation, case.input_shape, helpers.DEVICE,
               target=case.target, callbacks=[recorder])
    recorder.iterates[0] = case.to_current_layout(fx["z0"])
    return fx, recorder, case


@pytest.mark.parametrize("case_name", [pytest.param(c, marks=delta(c)) for c in fixtures.cases("trajectory")])
def test_trajectory_matches_v1(case_name: str) -> None:
    fx, ours, case = _replay(case_name)
    assert len(ours.losses) == len(fx["losses"])
    lbfgs = case.attack.stages[0].driver.optimizer_cls is torch.optim.LBFGS
    last_step = LBFGS_COMPARABLE_STEPS if lbfgs else EARLY
    for k, t in enumerate(fx["steps"]):
        if t <= last_step:
            torch.testing.assert_close(ours.iterates[t], case.to_current_layout(fx["iterates"][k]), **ITERATE_TOL,
                                       msg=f"iterate z_{t}")
    n_early = last_step + 1 if lbfgs else EARLY + 1
    torch.testing.assert_close(torch.tensor(ours.losses[:n_early]), torch.tensor(fx["losses"][:n_early]),
                               rtol=EARLY_LOSS_RTOL, atol=0.0, msg="early losses")
    if not lbfgs:
        torch.testing.assert_close(torch.tensor(ours.losses), torch.tensor(fx["losses"]), rtol=1e-3, atol=0.0,
                                   msg="loss trajectory")


def test_stg_matches_v1_with_v1_defects_replicated() -> None:
    """Evidence for the INTENTIONAL traj-see_through_gradients delta: only v1's two defects differ."""
    fx = fixtures.load("traj-see_through_gradients")
    case = v1_equivalent.with_v1_defects(v1_equivalent.trajectory_case(fx))
    ours = _Trajectory()
    run_attack(case.attack, case.model, case.observation, case.input_shape, helpers.DEVICE, target=case.target,
               callbacks=[ours])
    ours.iterates[0] = case.to_current_layout(fx["z0"])
    for k, t in enumerate(fx["steps"]):
        if t <= EARLY:
            torch.testing.assert_close(ours.iterates[t], case.to_current_layout(fx["iterates"][k]), **ITERATE_TOL)
    torch.testing.assert_close(torch.tensor(ours.losses), torch.tensor(fx["losses"]), rtol=1e-3, atol=0.0)
