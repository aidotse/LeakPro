#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Modular gradient-inversion attacks: compose an ``Attack`` from a task, stages and components, then run it.

``presets`` holds ready-made attacks, ``components`` the building blocks (losses, drivers, representations,
modalities, targets, evaluators, objectives). See the README next to this file.
"""
from leakpro.attacks.gia_attacks.modular.core.attack import Attack, run_attack
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.simulate import simulate_client
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, RunMetadata, State, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel

__all__ = [
    "Attack", "ClientObservation", "Context", "FedAvg", "FedSGD", "OptimStage", "RunMetadata", "State", "TargetInfo", "Task",
    "ThreatModel", "run_attack", "simulate_client",
]
