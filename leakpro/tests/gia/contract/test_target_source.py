#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""TargetSource contract: binding copies, TargetInfo output, axis slicing, differentiable parameters."""
from __future__ import annotations

import pytest
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import TargetSource
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, TargetInfo
from leakpro.tests.gia.contract.conftest import assert_unchanged, make_ctx, seed_axes, snapshot
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(TargetSource))


def _ctx(case):  # noqa: ANN001, ANN202
    axes = seed_axes() if case.supports_axes else AxisManager()
    return make_ctx(axes=axes, observation=ClientObservation(), **case.ctx_kwargs)


def test_bind_returns_a_copy_and_leaves_the_configuration_unchanged(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    before = snapshot(case.component)
    bound = case.component.bind(case.state(), _ctx(case))
    # A source holding no run state may return itself (the interface default).
    assert bound is not case.component or not vars(case.component)
    assert_unchanged(case.component, before)


def test_current_is_a_target_info(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    assert isinstance(case.component.bind(case.state(), _ctx(case)).current(), TargetInfo)


def test_integer_slice_removes_one_axis_dimension(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    target = case.component.bind(case.state(), _ctx(case)).current()
    sliced = case.component.slice(target, 0, 0)
    if target.axis_ndim == 0:
        assert sliced.axis_ndim == 0
        return
    assert sliced.axis_ndim == target.axis_ndim - 1
    for name, value in target.fields().items():
        if isinstance(value, Tensor):
            assert tuple(sliced.fields()[name].shape) == tuple(value.shape[1:]), name


def test_parameters_require_grad(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    for group in case.component.bind(case.state(), _ctx(case)).parameters():
        for p in group["params"]:
            assert p.requires_grad and p.is_leaf, group["name"]


def test_a_source_without_axis_support_rejects_an_axis_prefix(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    if case.supports_axes:
        pytest.skip("supports the axis prefix")
    ctx = make_ctx(axes=seed_axes(), observation=ClientObservation(), **case.ctx_kwargs)
    with pytest.raises(ValueError, match="axis"):
        case.component.bind(case.state(), ctx)
