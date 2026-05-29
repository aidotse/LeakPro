#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
# ruff: noqa: F401
"""Import all component modules so their @register decorators populate the registry.

This module is imported as a side effect by ``config/__init__.py``.  Add one line
here for every new component module that uses ``@register``.
"""

from leakpro.attacks.gia_attacks.modular.components import (
    initialization,
    label_inference,
    transition_strategies,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks import (
    bn_statistics_strategies,
    consensus_strategies,
    constraints,
    label_strategies,
    loss_components,
    mean_strategies,
    noise_strategies,
    optimizer_factories,
    step_strategies,
)
