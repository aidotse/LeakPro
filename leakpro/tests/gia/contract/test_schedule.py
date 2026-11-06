#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Schedule contract: the built scheduler steps through a run of any length with a valid learning rate."""
from __future__ import annotations

import math

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.interfaces import Schedule
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(Schedule))


@pytest.mark.parametrize("max_iter", [1, 2, 7, 50])
def test_build_steps_for_max_iter_steps(cls, factory, max_iter) -> None:  # noqa: ANN001
    schedule = build(cls, factory).component
    param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = schedule.build(optimizer, max_iter)
    for _ in range(max_iter):
        param.grad = torch.ones_like(param)
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert math.isfinite(lr) and lr >= 0.0
