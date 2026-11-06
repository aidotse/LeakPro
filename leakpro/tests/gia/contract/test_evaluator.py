#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Evaluator contract: at ground truth every metric is a finite number and takes its perfect value."""
from __future__ import annotations

import math

import pytest

from leakpro.attacks.gia_attacks.modular.core.evaluation import EvalResult
from leakpro.attacks.gia_attacks.modular.core.interfaces import Evaluator
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(Evaluator))


def test_metrics_at_ground_truth(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    result = case.component.evaluate(case.recon, case.target, state=case.state)
    assert isinstance(result, EvalResult)
    assert case.expected.keys() <= result.metrics.keys()
    for name, value in result.metrics.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), name
        expected = case.expected.get(name)
        if callable(expected):
            assert expected(value), f"{name}={value}"
            continue
        assert math.isfinite(value), f"{name}={value}"
        if expected is not None:
            assert value == pytest.approx(expected, abs=1e-4), name
