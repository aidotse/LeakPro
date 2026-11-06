#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Representations (parameterizations of the reconstruction) for the modular gradient-inversion framework."""
from leakpro.attacks.gia_attacks.modular.components.representations.diffusion import DiffusionRepresentation
from leakpro.attacks.gia_attacks.modular.components.representations.fixed import FixedRepresentation
from leakpro.attacks.gia_attacks.modular.components.representations.gan import (
    GANRepresentation,
    UnfrozenGANRepresentation,
)
from leakpro.attacks.gia_attacks.modular.components.representations.gifd import GIFDIntermediateRepresentation
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation

__all__ = [
    "DiffusionRepresentation",
    "FixedRepresentation",
    "GANRepresentation",
    "GIFDIntermediateRepresentation",
    "PixelRepresentation",
    "UnfrozenGANRepresentation",
]
