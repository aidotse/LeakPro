#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Run-scoped mutable resources, so configuration objects never carry state from one run to the next."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, nn

    from leakpro.attacks.gia_attacks.modular.core.state import Context


@dataclass
class RunSession:
    """Mutable resources of one run; created by run_attack, never shared between runs.

    Attributes:
        execution_model: The model the attacker simulates the client on in the current stage, a
            private copy of ``ctx.model`` in the task's module mode.
        pristine_params: The server weights the simulation starts from (``ctx.model``'s parameters).
        pristine_buffers: The server buffers the simulation starts from (``ctx.model``'s buffers).

    """

    execution_model: nn.Module | None = None
    pristine_params: dict[str, Tensor] = field(default_factory=dict)
    pristine_buffers: dict[str, Tensor] = field(default_factory=dict)
    # id(original) -> (original, copy); holding the original keeps its id from being reused.
    _owned: dict[int, tuple[nn.Module, nn.Module]] = field(default_factory=dict, repr=False)

    def begin_stage(self, ctx: Context) -> None:
        """Fresh execution model for the stage: deepcopy(ctx.model), prepared by ctx.task.objective in ctx.task.model_mode."""
        model = copy.deepcopy(ctx.model)
        ctx.task.objective.prepare_model(model, ctx.task.model_mode)
        self.execution_model = model
        self.pristine_params = dict(ctx.model.named_parameters())
        self.pristine_buffers = dict(ctx.model.named_buffers())

    def owned_copy(self, module: nn.Module) -> nn.Module:
        """A private deep copy of a caller-owned module, made once per run and reused by later stages."""
        key = id(module)
        if key not in self._owned:
            self._owned[key] = (module, copy.deepcopy(module))
        return self._owned[key][1]
