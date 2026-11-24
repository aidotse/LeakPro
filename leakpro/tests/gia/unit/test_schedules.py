#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import math

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import Cosine, LinearDecay, MultiStep


def _lrs(schedule, max_iter=8):
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=1.0)
    sch = schedule.build(opt, max_iter)
    out = []
    for _ in range(max_iter):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    return out


def test_multistep_with_explicit_milestones() -> None:
    assert _lrs(MultiStep(milestones=(2, 5), gamma=0.5)) == [1, 1, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25]


def test_multistep_fractions_default_to_three_five_seven_eighths() -> None:
    lrs = _lrs(MultiStep(), max_iter=8)
    assert lrs[2] == 1 and lrs[3] == pytest.approx(0.1) and lrs[5] == pytest.approx(0.01) and lrs[7] == pytest.approx(0.001)


def test_multistep_divisors_floor_the_division_like_python_floats() -> None:
    p = torch.zeros(1, requires_grad=True)
    sch = MultiStep(divisors=(2, 1.6, 1.142)).build(torch.optim.SGD([p], lr=1.0), 4800)
    assert sorted(sch.milestones) == [4800 // 2, int(4800 // 1.6), int(4800 // 1.142)]
    assert 4800 // 1.6 == 2999


def test_every_schedule_builds_with_any_max_iter() -> None:   # regression: only cosine used to accept T_max
    for schedule in (Cosine(), MultiStep(), LinearDecay()):
        assert len(_lrs(schedule, max_iter=5)) == 5


def test_cosine_follows_the_half_cosine_from_base_lr_to_eta_min() -> None:
    lrs = _lrs(Cosine(), max_iter=8)
    assert lrs[0] == 1.0 and lrs[4] == pytest.approx(0.5) and lrs[2] == pytest.approx(0.5 * (1 + 2 ** -0.5))
    assert lrs[7] == pytest.approx(0.5 * (1 + math.cos(7 * math.pi / 8)))
    floor = _lrs(Cosine(eta_min=0.2), max_iter=8)
    assert floor[4] == pytest.approx(0.2 + 0.4)


def test_linear_decay_interpolates_between_start_and_end_factor() -> None:
    lrs = _lrs(LinearDecay(start_factor=1.0, end_factor=0.1), max_iter=8)
    assert lrs[0] == pytest.approx(1.0)
    assert lrs[4] == pytest.approx(1.0 - 0.9 * 4 / 8)
    assert lrs[7] == pytest.approx(1.0 - 0.9 * 7 / 8)
    # Having run all steps the factor is the end factor.
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=1.0)
    sch = LinearDecay(end_factor=0.1).build(opt, 8)
    for _ in range(8):
        opt.step()
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
