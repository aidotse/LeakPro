#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Initializer contract: a known space, and samples of the requested shape on the requested device."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.interfaces import SPACES, Initializer
from leakpro.tests.gia import helpers
from leakpro.tests.gia.contract.registry import INIT_SHAPE, build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(Initializer))


def test_sample_has_the_requested_shape_and_device(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    assert case.component.space in SPACES
    if not case.draws:
        # Documented: the representation takes its start from the state instead.
        with pytest.raises(ValueError, match="state"):
            case.component.sample(INIT_SHAPE, helpers.DEVICE)
        return
    sample = case.component.sample(INIT_SHAPE, helpers.DEVICE)
    assert tuple(sample.shape) == INIT_SHAPE
    assert sample.device.type == helpers.DEVICE.type
    assert not sample.requires_grad


def test_a_seed_fixes_the_draw(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    if case.seeded is None:
        pytest.skip("draws no random numbers")
    a, b, c = (case.seeded(seed).sample(INIT_SHAPE, helpers.DEVICE) for seed in (4, 4, 5))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
