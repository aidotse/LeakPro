#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Threat model system for gradient inversion attacks.

The ThreatModel is a data firewall: its ``sanitize()`` method strips fields
from a ``ClientObservation`` that the attacker is not allowed to see before
the observation reaches any LossTerm. Components declare the fields they read
(``requirements()``), and ``run_attack`` checks them against the sanitized copy
before the first stage runs.

Usage::

    attack.threat_model = ThreatModel(has_bn_statistics=True, has_local_hyperparameters=True)
    state, ctx = run_attack(attack, model, observation, ...)  # sanitizes a copy of observation
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation


@dataclass
class ThreatModel:
    """Explicit capability flags that define what the attacker can observe.

    Set a flag to ``True`` to grant access; ``False`` (the default) causes
    ``sanitize()`` to strip the corresponding ``ClientObservation`` field.

    Attributes:
        has_bn_statistics:          BN running statistics (mean/var per layer), before and after
            local training.
        has_local_hyperparameters:  Local training config (batch size, loss value).

    """

    has_bn_statistics: bool = False
    has_local_hyperparameters: bool = False

    def sanitize(self, obs: ClientObservation) -> ClientObservation:
        """A copy of ``obs`` without the fields the attacker is not allowed to see; ``obs`` is unchanged.

        Typed fields are set to None; ``extra`` keys mapped to a withheld flag are deleted, so a component
        that reads one fails with a KeyError rather than computing on a None. Tensors are shared with
        ``obs``, not copied.
        """
        stripped: dict = {}
        if not self.has_bn_statistics:
            stripped.update(bn_stats=None, pre_bn_stats=None)
        if not self.has_local_hyperparameters:
            stripped.update(batch_size=None, loss_value=None)
        extra = dict(obs.extra)
        for flag, keys in _EXTRA_KEYS_BY_FLAG.items():
            if not getattr(self, flag):
                for key in keys:
                    extra.pop(key, None)
        return replace(obs, extra=extra, **stripped)

    def granted(self) -> list[str]:
        """Names of the capabilities this threat model grants — for logging/reporting."""
        return [f.name for f in fields(self) if getattr(self, f.name)]


# Flag -> keys stripped from ClientObservation.extra when the flag is False.
# Declared as data so adding a capability is a one-line change, and kept
# module-level so it stays out of the dataclass field list and __eq__/__repr__.
_EXTRA_KEYS_BY_FLAG: dict[str, tuple[str, ...]] = {}
