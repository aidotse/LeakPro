#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""LossTerm contract: unchanged configuration, finite scalar values, declared requirements."""
from __future__ import annotations

import dataclasses

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.interfaces import LossTerm
from leakpro.attacks.gia_attacks.modular.core.loss_eval import evaluate_stage_losses
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossScope
from leakpro.attacks.gia_attacks.modular.core.requirements import REQUIREMENTS
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context
from leakpro.tests.gia import helpers
from leakpro.tests.gia.contract.conftest import assert_unchanged, client_target, make_ctx, snapshot
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(LossTerm))

_OBSERVATION_FIELDS = ("gradients", "bn_stats", "pre_bn_stats", "batch_size", "loss_value")


def _evaluate(term: LossTerm, ctx: Context, recon: torch.Tensor) -> torch.Tensor:
    """The term's weighted value at ``recon`` through the shared evaluator, on a stage-bound copy."""
    ctx.session.begin_stage(ctx)
    bound = term.bind(ctx)
    try:
        scope = getattr(term, "scope", LossScope.PER_SLICE)
        out = evaluate_stage_losses({"term": (bound, 1.0)}, ctx, recon, client_target(), scope=scope,
                                    seed_index=None if scope is LossScope.GROUP else 0, full_recon=recon)
        return out.terms["term"]
    finally:
        bound.release()


def _recon(ctx: Context) -> torch.Tensor:
    """A random data-space iterate shaped ``[*axes, *input_shape]``, requiring grad like a driver's."""
    return torch.rand(*ctx.axes.shape, *ctx.input_shape, generator=torch.Generator().manual_seed(5)).requires_grad_(True)


def test_bind_leaves_the_configuration_unchanged(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    ctx = make_ctx(**case.ctx_kwargs)
    before = snapshot(case.component)
    ctx.session.begin_stage(ctx)
    bound = case.component.bind(ctx)
    bound.release()
    assert_unchanged(case.component, before)


def test_compute_is_a_finite_scalar(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    ctx = make_ctx(**case.ctx_kwargs)
    value = _evaluate(case.component, ctx, _recon(ctx))
    assert value.dim() == 0
    assert torch.isfinite(value)


def test_requirements_are_known(cls, factory) -> None:  # noqa: ANN001
    requirements = build(cls, factory).component.requirements()
    assert isinstance(requirements, frozenset)
    assert requirements <= set(REQUIREMENTS)


def test_gradient_matching_is_zero_at_the_client_batch(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    if not case.matches_gradients:
        pytest.skip("not a gradient-matching term")
    ctx = make_ctx(**case.ctx_kwargs)
    value = _evaluate(case.component, ctx, helpers.client_batch()[0])
    assert value.item() == pytest.approx(0.0, abs=1e-5)


def test_undeclared_observation_fields_are_not_read(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    full = make_ctx(**case.ctx_kwargs)
    declared = case.component.requirements()
    stripped: ClientObservation = dataclasses.replace(
        full.observation, **{f: None for f in _OBSERVATION_FIELDS if f not in declared})
    ctx = dataclasses.replace(full, observation=stripped)
    recon = _recon(ctx)
    assert torch.equal(_evaluate(case.component, ctx, recon), _evaluate(case.component, full, recon))
