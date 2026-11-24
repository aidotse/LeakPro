#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Callback that logs per-step loss and per-stage timing."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback
from leakpro.utils.logger import logger

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.events import StepEvent
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class ConsoleLogger(Callback):
    """Logs iteration, total loss, and optional per-component breakdown every log_freq steps.

    If track_time=True (default), on_stage_start records a timestamp so that
    on_stage_end can report total elapsed seconds for the stage.
    """

    def __init__(self, log_freq: int = 50, track_time: bool = True) -> None:
        self.log_freq = log_freq
        self.track_time = track_time
        self.stage_start_times: dict[int, float] = {}

    def on_stage_start(self, stage_idx: int, state: State, ctx: Context) -> None:  # noqa: ARG002 -- interface signature
        """Logs the stage banner and records the start time."""
        if self.track_time:
            self.stage_start_times[stage_idx] = time.time()
        logger.info(f"\n{'='*40}\nStarting Stage {stage_idx}...\n{'='*40}")

    def on_step_end(self, event: StepEvent) -> None:
        """Logs the loss and loss components every ``log_freq`` steps."""
        if event.step % self.log_freq == 0:
            msg = f"Iter {event.step:04d} | Loss: {event.loss:10.4f}"
            if event.components:
                msg += " | " + " | ".join(f"{k}: {v:10.4f}" for k, v in event.components.items())
            logger.info(msg)

    def on_stage_end(self, stage_idx: int, state: State, ctx: Context) -> None:  # noqa: ARG002 -- interface signature
        """Logs the stage's elapsed time and final loss."""
        elapsed = 0.0
        if self.track_time and stage_idx in self.stage_start_times:
            elapsed = time.time() - self.stage_start_times[stage_idx]
        logger.info(f"Stage {stage_idx} Complete in {elapsed:.2f}s | Final Loss: {state.loss:.4f}")
