#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Projection constraints for GAN-based representations (GIFD, spherical latent search)."""
from leakpro.attacks.gia_attacks.modular.components.constraints.l1_ball import project_l1_ball
from leakpro.attacks.gia_attacks.modular.components.constraints.spherical import project_to_radius

__all__ = [
    "project_l1_ball",
    "project_to_radius",
]
