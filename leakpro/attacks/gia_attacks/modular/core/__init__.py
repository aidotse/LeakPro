#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Core interfaces and state for the modular gradient-inversion framework.

The public extension surface: the interfaces a new component implements, the inputs a loss term receives,
the parallel axes of a multi-seed or multi-epoch attack, and state checkpointing. Running and composing an
attack uses the names exported by ``leakpro.attacks.gia_attacks.modular``.
"""
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.evaluation import EvalResult
from leakpro.attacks.gia_attacks.modular.core.interfaces import (
    Aggregator,
    BatchSampler,
    Callback,
    ClientProtocol,
    Driver,
    Evaluator,
    GradTransform,
    Initializer,
    LossTerm,
    Modality,
    Objective,
    Representation,
    Schedule,
    TargetInference,
    TargetSource,
)
from leakpro.attacks.gia_attacks.modular.core.io import load_state, save_state
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs, LossScope

__all__ = [
    "Aggregator", "Axis", "AxisManager", "BatchSampler", "Callback", "ClientProtocol", "Driver", "EvalResult", "Evaluator",
    "GradTransform", "Initializer", "LossInputs", "LossScope", "LossTerm", "Modality", "Objective", "Representation",
    "Schedule", "TargetInference", "TargetSource", "load_state", "save_state",
]
