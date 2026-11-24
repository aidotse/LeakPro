#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for core/meta.py.

Tests:
  1. MetaSGD.step(create_graph=True) keeps the second-order graph alive (updated
     parameters trace back through the loss to the loss input).
  2. MetaSGD.step(create_graph=False) drops the second-order graph (updated
     parameters do NOT trace back to the loss input — VRAM-safe).
  3. MetaAdam.step(create_graph=False) also drops the graph.
  4. MetaAdam.step(create_graph=True) keeps the graph.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


from leakpro.attacks.gia_attacks.modular.core.meta import MetaAdam, MetaSGD


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_param_and_loss() -> tuple[nn.Parameter, torch.Tensor, torch.Tensor]:
    """Return (param, x, loss) where loss = (param * x).sum()."""
    param = nn.Parameter(torch.tensor([1.0, 2.0]))
    x = torch.tensor([3.0, 4.0], requires_grad=True)
    loss = (param * x).sum()
    return param, x, loss


# ── MetaSGD graph propagation ─────────────────────────────────────────────────

class TestMetaSGDGraphPropagation:
    """Verify that create_graph controls whether second-order gradients are available."""

    def test_create_graph_true_keeps_path_to_loss_input(self):
        """With create_graph=True, updated params trace back through the loss to x."""
        param, x, loss = _make_param_and_loss()
        sgd = MetaSGD(lr=0.1, foreach=False)
        updated = sgd.step(loss, OrderedDict([("p", param)]), create_graph=True)

        test_loss = updated["p"].sum()
        g_x = torch.autograd.grad(test_loss, x, allow_unused=True)[0]
        assert g_x is not None, "create_graph=True should preserve path through loss to x"

    def test_create_graph_false_drops_path_to_loss_input(self):
        """With create_graph=False, updated params do NOT trace back to x (VRAM-safe)."""
        param, x, loss = _make_param_and_loss()
        sgd = MetaSGD(lr=0.1, foreach=False)
        updated = sgd.step(loss, OrderedDict([("p", param)]), create_graph=False)

        # grad was computed without create_graph, so it is detached from the loss graph.
        # updated["p"] depends on param (leaf) but not on x.
        test_loss = updated["p"].sum()
        g_x = torch.autograd.grad(test_loss, x, allow_unused=True)[0]
        assert g_x is None, "create_graph=False must drop the second-order path to x"

    def test_create_graph_false_updated_params_still_require_grad(self):
        """Updated params still require_grad (they depend on the leaf param)."""
        param, x, loss = _make_param_and_loss()
        sgd = MetaSGD(lr=0.1, foreach=False)
        updated = sgd.step(loss, OrderedDict([("p", param)]), create_graph=False)
        assert updated["p"].requires_grad


# ── MetaAdam graph propagation ────────────────────────────────────────────────

class TestMetaAdamGraphPropagation:
    def test_create_graph_false_drops_path_to_loss_input(self):
        """MetaAdam with create_graph=False does not retain the loss→input graph."""
        param, x, loss = _make_param_and_loss()
        adam = MetaAdam(lr=0.1)
        updated = adam.step(loss, OrderedDict([("p", param)]), create_graph=False)

        test_loss = updated["p"].sum()
        g_x = torch.autograd.grad(test_loss, x, allow_unused=True)[0]
        assert g_x is None, "MetaAdam create_graph=False must drop second-order graph"

    def test_create_graph_true_keeps_path_to_loss_input(self):
        param, x, loss = _make_param_and_loss()
        adam = MetaAdam(lr=0.1)
        updated = adam.step(loss, OrderedDict([("p", param)]), create_graph=True)

        test_loss = updated["p"].sum()
        g_x = torch.autograd.grad(test_loss, x, allow_unused=True)[0]
        assert g_x is not None, "MetaAdam create_graph=True should keep second-order graph"
