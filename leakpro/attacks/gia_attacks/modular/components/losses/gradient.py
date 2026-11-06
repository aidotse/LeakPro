#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Gradient-matching loss terms: distance metrics between dummy and target gradients."""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 -- conventional alias
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import LossTerm
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs
from leakpro.attacks.gia_attacks.modular.core.requirements import Requirement


class CosineGradientLoss(LossTerm):
    """Cosine-distance gradient matching (Geiping 2020).

    Globally flattens and concatenates all per-layer gradients before computing
    the cosine distance so every parameter element has equal weight, matching the
    paper's formulation.  The driver supplies the pre-computed dummy_obs.
    """

    def requirements(self) -> frozenset[Requirement]:
        """The observed gradients it matches."""
        return frozenset({"gradients"})

    def compute(self, inputs: LossInputs) -> Tensor:
        """Cosine distance between the flattened dummy and observed gradients."""
        dummy_flat = torch.cat([g.flatten() for g in inputs.dummy_obs.gradients])
        obs_flat = torch.cat([g.flatten() for g in inputs.target_obs.gradients])
        return 1.0 - F.cosine_similarity(dummy_flat, obs_flat, dim=0)


class L2GradientLoss(LossTerm):
    """L2 (MSE) gradient matching loss used in DLG / iDLG."""

    def requirements(self) -> frozenset[Requirement]:
        """The observed gradients it matches."""
        return frozenset({"gradients"})

    def compute(self, inputs: LossInputs) -> Tensor:
        """Sum over parameters of the squared error between the dummy and observed gradients."""
        loss = torch.tensor(0.0, device=inputs.ctx.device)
        for dg, tg in zip(inputs.dummy_obs.gradients, inputs.target_obs.gradients):
            loss = loss + F.mse_loss(dg, tg, reduction="sum")
        return loss
