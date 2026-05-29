#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Stage ABC — the unit of execution in a multi-stage GIA attack."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.callbacks import Callback
    from leakpro.attacks.gia_attacks.modular.core.state import RunContext, WorkingState


class Stage(ABC):
    """Abstract base class for a single optimization stage.

    A stage receives the current WorkingState and immutable RunContext, runs
    some optimization process, and returns the (possibly mutated) WorkingState.

    Mutation convention: stages may mutate ``state`` in place and return the
    same instance — or return a fresh instance. The orchestrator always uses
    the returned value.

    Standalone execution::

        stage = orch.build_stage(index=1)
        state = WorkingState(reconstruction=my_tensor, labels=my_labels)
        ctx   = RunContext(target_model=..., client_observations=..., ...)
        state = stage.run(state, ctx, stage_idx=1, callbacks=[])
    """

    @abstractmethod
    def run(
        self,
        state: "WorkingState",
        ctx: "RunContext",
        *,
        stage_idx: int = 0,
        callbacks: "list[Callback]",
    ) -> "WorkingState":
        """Run this stage and return the updated state.

        Args:
            state:      Current WorkingState (may be mutated in place).
            ctx:        Immutable RunContext for this attack run.
            stage_idx:  Index of this stage in the multi-stage sequence (used
                        in callback dispatch so hooks can distinguish stages).
            callbacks:  Callbacks to fire on each step and at stage boundaries.

        Returns:
            Updated WorkingState (may be the same object or a new one).
        """


__all__ = ["Stage"]
