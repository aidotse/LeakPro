#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Training objectives: how a client model consumes a batch and yields its training loss."""
from leakpro.attacks.gia_attacks.modular.components.objectives.classification import (
    BinaryCrossEntropy,
    CrossEntropy,
    MeanSquaredError,
)

__all__ = [
    "BinaryCrossEntropy",
    "CrossEntropy",
    "MeanSquaredError",
]
