#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The framework objective, configured like the v1 preset, has v1's terms, value and descent direction.

Points are in data space; v1 optimizes the normalized input, and its recorded gradient was mapped
back to data space with the chain rule, so the two gradients are directly comparable.
"""
import pytest
import torch
import torch.nn.functional as F

from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.tests.gia import helpers
from leakpro.tests.gia.parity import fixtures, v1_equivalent
from leakpro.tests.gia.parity.deltas import delta


def _evaluate(case: str, point: int) -> tuple[dict[str, float], float, torch.Tensor, dict]:
    fx = fixtures.load(case)
    stage = v1_equivalent.stage(fx["settings"]["config"])
    ctx = v1_equivalent.context(fx)
    target = TargetInfo(value=v1_equivalent.target_labels(fx))
    x = fx["points"][point].clone().requires_grad_(True)
    terms = helpers.stage_terms(stage, ctx, x, target)
    total = sum(terms.values())
    (grad,) = torch.autograd.grad(total, x)
    return {name: value.item() for name, value in terms.items()}, total.item(), grad, fx


def _term_cases() -> list:
    params = []
    for case in fixtures.cases("loss_points"):
        for term in fixtures.load(case)["terms"][0]:
            test_id = f"{case}:{term}"
            params.append(pytest.param(case, term, id=test_id, marks=delta(test_id)))
    return params


@pytest.mark.parametrize(("case", "term"), _term_cases())
def test_each_term_matches_v1(case: str, term: str) -> None:
    for point in range(3):
        ours, _, _, fx = _evaluate(case, point)
        assert ours[term] == pytest.approx(fx["terms"][point][term], rel=1e-4, abs=1e-7), f"point {point}"


@pytest.mark.parametrize("case", [pytest.param(c, marks=delta(c)) for c in fixtures.cases("loss_points")])
def test_objective_value_and_direction_match_v1(case: str) -> None:
    for point in range(3):
        ours, total, grad, fx = _evaluate(case, point)
        assert set(ours) == set(fx["terms"][point]), "term sets differ"
        assert total == pytest.approx(fx["total"][point].item(), rel=1e-4, abs=1e-7), f"point {point} value"
        cosine = F.cosine_similarity(grad.flatten(), fx["grad_wrt_data"][point].flatten(), dim=0).item()
        assert cosine > 0.999, f"point {point} direction (cosine {cosine:.4f})"
