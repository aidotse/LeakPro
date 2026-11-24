#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Components for the modular gradient-inversion framework."""

from leakpro.attacks.gia_attacks.modular.components.aggregators import (
    BestLossAggregator,
    ChainAggregator,
    EpochMatchingAggregator,
    MeanAggregator,
)
from leakpro.attacks.gia_attacks.modular.components.initializers import FromState, FromTensor, Gaussian, Uniform
from leakpro.attacks.gia_attacks.modular.components.loggers import ConsoleLogger
from leakpro.attacks.gia_attacks.modular.components.samplers import RandomSampler, SequentialSampler
from leakpro.attacks.gia_attacks.modular.components.target_inference import (
    DimitrovInference,
    GengInference,
    IDLGInference,
    ZeroLabelInference,
)

__all__ = [
    "BestLossAggregator",
    "ChainAggregator",
    "ConsoleLogger",
    "DimitrovInference",
    "EpochMatchingAggregator",
    "FromState",
    "FromTensor",
    "Gaussian",
    "GengInference",
    "IDLGInference",
    "MeanAggregator",
    "RandomSampler",
    "SequentialSampler",
    "Uniform",
    "ZeroLabelInference",
]
