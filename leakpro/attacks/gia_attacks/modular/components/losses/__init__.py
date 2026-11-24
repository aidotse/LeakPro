#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Composable loss terms for gradient matching and image or multi-run regularization.

Every loss is re-exported here, so ``from leakpro...components.losses import CosineGradientLoss`` works.
"""
from leakpro.attacks.gia_attacks.modular.components.losses.gradient import (
    CosineGradientLoss,
    L2GradientLoss,
)
from leakpro.attacks.gia_attacks.modular.components.losses.regularization import (
    BNStatsLoss,
    EpochOrderInvariantPrior,
    GroupConsistencyLoss,
    L2ImageNormLoss,
    TVLoss,
)

__all__ = [
    "CosineGradientLoss",
    "L2GradientLoss",
    "TVLoss",
    "BNStatsLoss",
    "L2ImageNormLoss",
    "GroupConsistencyLoss",
    "EpochOrderInvariantPrior",
]
