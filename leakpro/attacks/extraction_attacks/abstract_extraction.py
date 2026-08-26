#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Shared lifecycle for standalone extraction attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from time import perf_counter

from leakpro.reporting.extraction_result import ExtractionResult


class AttackState(str, Enum):
    """Lifecycle states shared by the standalone attacks."""

    CREATED = "created"
    PREPARING = "preparing"
    PREPARED = "prepared"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AbstractExtraction(ABC):
    """LeakPro-style interface for a training-data extraction attack."""

    state: AttackState

    def _initialize_trace(self) -> None:
        """Initialize a compact execution trace for audit reproduction."""
        self.execution_trace: list[dict[str, object]] = []
        self._trace_start = perf_counter()

    def _record_trace(self, phase: str, **details: object) -> None:
        """Record a phase boundary without storing model inputs or image data."""
        self.execution_trace.append(
            {
                "attack_id": getattr(self, "attack_id", "unknown"),
                "phase": phase,
                "elapsed_seconds": round(perf_counter() - self._trace_start, 6),
                **details,
            }
        )

    @abstractmethod
    def description(self) -> dict[str, str]:
        """Return the paper reference, threat model, and implementation scope."""

    @abstractmethod
    def prepare_attack(self) -> None:
        """Validate inputs and prepare reusable attack state."""

    @abstractmethod
    def run_attack(self) -> ExtractionResult:
        """Run the attack and return candidates plus audit metrics."""

    def _execute_once(self, operation: Callable[[], ExtractionResult]) -> ExtractionResult:
        """Run one prepared attack and make every outcome terminal."""
        if self.state is not AttackState.PREPARED:
            raise RuntimeError("A prepared extraction attack can run only once.")
        self.state = AttackState.RUNNING
        try:
            result = operation()
        except BaseException:
            self.state = AttackState.FAILED
            raise
        self.state = AttackState.COMPLETED
        return result

    def _prepare_once(self, operation: Callable[[], None]) -> None:
        """Prepare one attack and make every outcome explicit and terminal."""
        self._begin_preparation()
        try:
            operation()
        except BaseException:
            self.state = AttackState.FAILED
            raise
        self.state = AttackState.PREPARED

    def _begin_preparation(self) -> None:
        """Claim this one-shot attack instance before preparation mutates it."""
        if self.state is not AttackState.CREATED:
            raise RuntimeError(
                "Extraction attack instances are one-shot; create a new LeakPro instance to run another audit."
            )
        self.state = AttackState.PREPARING
