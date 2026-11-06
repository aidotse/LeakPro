#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Representation contract: binding copies, data-space decode with the axis prefix, detached outputs, resume."""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import Representation
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, State
from leakpro.tests.gia.contract.conftest import assert_unchanged, make_ctx, seed_axes, snapshot
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(Representation))


def _bound(cls, factory):  # noqa: ANN001, ANN202
    case = build(cls, factory)
    ctx = make_ctx(axes=seed_axes(), observation=ClientObservation(), **case.ctx_kwargs)
    state = case.state(ctx)
    return case, ctx, state, case.component.bind(state, ctx)


def _params(rep: Representation) -> list[Tensor]:
    return [p for group in rep.parameters() for p in group["params"]]


def test_bind_returns_a_copy_and_leaves_the_configuration_unchanged(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    ctx = make_ctx(axes=seed_axes(), observation=ClientObservation(), **case.ctx_kwargs)
    before = snapshot(case.component)
    bound = case.component.bind(case.state(ctx), ctx)
    assert bound is not case.component
    assert_unchanged(case.component, before)


def test_decode_has_the_axis_prefix_and_input_shape(cls, factory) -> None:  # noqa: ANN001
    _, ctx, _, rep = _bound(cls, factory)
    assert tuple(rep.decode().shape) == (*ctx.axes.shape, *ctx.input_shape)


def test_image_decode_stays_in_data_space_after_project(cls, factory) -> None:  # noqa: ANN001
    case, ctx, _, rep = _bound(cls, factory)
    if not case.image:
        pytest.skip("not an image representation")
    with torch.no_grad():
        for p in _params(rep):
            p.add_(3.0 * torch.randn_like(p))
    rep.project(ctx)
    observable = rep.to_observable()
    assert observable.min() >= 0.0 and observable.max() <= 1.0
    if case.bounded:
        with torch.no_grad():
            decoded = rep.decode()
        assert decoded.min() >= 0.0 and decoded.max() <= 1.0


def test_observable_and_artifacts_are_detached_and_carry_the_prefix(cls, factory) -> None:  # noqa: ANN001
    _, ctx, _, rep = _bound(cls, factory)
    observable = rep.to_observable()
    assert not observable.requires_grad and observable.grad_fn is None
    assert tuple(observable.shape) == (*ctx.axes.shape, *ctx.input_shape)
    tensors = {k: v for k, v in rep.artifacts().items() if isinstance(v, Tensor)}
    assert tensors, "a bound representation exposes its iterate as artifacts"
    for name, value in tensors.items():
        assert not value.requires_grad and value.grad_fn is None, name
        assert tuple(value.shape[: len(ctx.axes.shape)]) == ctx.axes.shape, name


def test_binding_from_its_own_artifacts_reproduces_decode(cls, factory) -> None:  # noqa: ANN001
    case, ctx, state, rep = _bound(cls, factory)
    with torch.no_grad():
        for p in _params(rep):
            p.add_(0.1 * torch.randn_like(p))
    resume = State(target=state.target, artifacts=rep.artifacts())
    if not case.resumes_from_artifacts:
        resume.input = rep.to_observable()
    resumed = case.component.bind(resume, ctx)
    with torch.no_grad():
        torch.testing.assert_close(resumed.decode(), rep.decode())


def test_parameters_are_leaves_requiring_grad(cls, factory) -> None:  # noqa: ANN001
    _, _, _, rep = _bound(cls, factory)
    for group in rep.parameters():
        assert {"name", "params"} <= group.keys()
        for p in group["params"]:
            assert p.requires_grad and p.is_leaf, group["name"]
