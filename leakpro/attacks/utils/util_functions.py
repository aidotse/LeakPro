#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Diverse util functions."""
from torch import Tensor, abs, mean


def total_variation(x: Tensor) -> Tensor:
        """Anisotropic TV."""
        dx = mean(abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = mean(abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy
