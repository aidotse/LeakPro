#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Dependency-free callback protocol for GIA attack observability.

Callbacks hook into the attack pipeline without coupling the framework to any
specific HPO or logging library. Implementations (MLflow, Optuna pruning, etc.)
live in the caller's codebase and import only this protocol.

Usage::

    class MyCallback(Callback):
        def on_step(self, stage_idx, state, ctx):
            print(f"stage {stage_idx} iter {state.iteration} loss {state.loss:.4f}")

    state, ctx = orch.run_attack(model, obs, callbacks=[MyCallback()])

To terminate a stage early (e.g. Optuna pruning), raise StopStage from on_step.
The orchestrator catches it, sets state.converged = False, and proceeds to the
next stage (or ends the attack if it was the last stage).
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import RunContext, WorkingState


class Callback(ABC):
    """Base class for attack pipeline callbacks.

    All methods have no-op defaults — only override the hooks you care about.
    Hooks receive both the mutable WorkingState and the immutable RunContext so
    callbacks can read any field without the framework threading individual args.
    """

    def on_attack_start(self, state: "WorkingState", ctx: "RunContext") -> None:
        """Called once after bootstrap (label inference + init), before stage 0."""

    def on_stage_start(self, stage_idx: int, state: "WorkingState", ctx: "RunContext") -> None:
        """Called at the beginning of each stage, before the first optimiser step."""

    def on_step(self, stage_idx: int, state: "WorkingState", ctx: "RunContext") -> None:
        """Called every log_interval steps inside a stage.

        Raise StopStage to terminate the current stage early.
        """

    def on_stage_end(self, stage_idx: int, state: "WorkingState", ctx: "RunContext") -> None:
        """Called after a stage completes (or is stopped early by StopStage)."""

    def on_attack_end(self, state: "WorkingState", ctx: "RunContext") -> None:
        """Called once after all stages have finished."""


class StopStage(Exception):
    """Raise from Callback.on_step to terminate the current stage early.

    The orchestrator catches this, sets state.converged = False, fires
    on_stage_end, and either continues to the next stage or returns.
    Useful for Optuna pruning: raise StopStage when trial.should_prune() is True.
    """


__all__ = ["Callback", "StopStage"]
