#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Canonical attack presets for the modular GIA framework.

Each module in this package provides one or more factory functions that return
a fully-configured :class:`~leakpro.attacks.gia_attacks.modular.config.schema.AttackConfig`
(and optionally a ``live_overrides`` dict for attacks that require pre-loaded
non-serialisable objects such as GAN generators).

Typical usage::

    from leakpro.attacks.gia_attacks.modular.presets import inverting_gradients_attack
    from leakpro.attacks.gia_attacks.modular.config.builder import AttackBuilder, resolve_attack_config

    cfg = inverting_gradients_attack(tv_weight=1e-3)
    cfg = resolve_attack_config(cfg, client_observations)
    orch = AttackBuilder.build(cfg)

For GAN-based attacks (GIAS, GIFD, GGL) the factory returns ``(config, live_overrides)``::

    from leakpro.attacks.gia_attacks.modular.presets import gias_attack

    cfg, overrides = gias_attack(huggingface_model="brownvc/R3GAN-CIFAR10")
    orch = AttackBuilder.build(cfg, live_overrides=overrides)
"""

from leakpro.attacks.gia_attacks.modular.presets.dimitrov import dimitrov_fedavg_attack
from leakpro.attacks.gia_attacks.modular.presets.dlg import dlg_attack
from leakpro.attacks.gia_attacks.modular.presets.ggdm import ggdm_attack
from leakpro.attacks.gia_attacks.modular.presets.ggl import ggl_attack
from leakpro.attacks.gia_attacks.modular.presets.gia_estimate import gia_estimate_attack
from leakpro.attacks.gia_attacks.modular.presets.gia_running import gia_running_attack
from leakpro.attacks.gia_attacks.modular.presets.gias import gias_attack
from leakpro.attacks.gia_attacks.modular.presets.gifd import gifd_attack
from leakpro.attacks.gia_attacks.modular.presets.gradinvdiff import gradinvdiff_attack
from leakpro.attacks.gia_attacks.modular.presets.huang import huang_attack
from leakpro.attacks.gia_attacks.modular.presets.idlg import idlg_attack
from leakpro.attacks.gia_attacks.modular.presets.inverting_gradients import inverting_gradients_attack
from leakpro.attacks.gia_attacks.modular.presets.see_through_gradients import see_through_gradients_attack

__all__ = [
    "dimitrov_fedavg_attack",
    "dlg_attack",
    "ggdm_attack",
    "ggl_attack",
    "gia_estimate_attack",
    "gia_running_attack",
    "gias_attack",
    "gifd_attack",
    "gradinvdiff_attack",
    "huang_attack",
    "idlg_attack",
    "inverting_gradients_attack",
    "see_through_gradients_attack",
]
