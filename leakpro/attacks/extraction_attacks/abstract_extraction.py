#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Common configuration and interface for extraction attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from leakpro.reporting.extraction_result import ExtractionResult


class ExtractionConfig(BaseModel):
    """Settings shared by extraction attacks."""

    random_seed: int = 42
    overwrite_results: bool = False
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AbstractExtraction(ABC):
    """LeakPro-style interface for a training-data extraction attack."""

    @abstractmethod
    def description(self) -> dict[str, str]:
        """Return the paper reference, threat model, and implementation scope."""

    @abstractmethod
    def prepare_attack(self) -> None:
        """Validate inputs and prepare attack data and models."""

    @abstractmethod
    def run_attack(self) -> ExtractionResult:
        """Run the attack and return candidates and metrics."""
