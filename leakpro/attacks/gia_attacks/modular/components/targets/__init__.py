#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Target sources for the modular gradient-inversion framework."""

from leakpro.attacks.gia_attacks.modular.components.targets.fixed import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.targets.joint import JointTarget
from leakpro.attacks.gia_attacks.modular.components.targets.null import NullTarget

__all__ = ["FixedTarget", "JointTarget", "NullTarget"]
