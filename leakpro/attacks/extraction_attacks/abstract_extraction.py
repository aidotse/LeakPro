#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Base class for training-data extraction attacks on causal language models.

Extraction attacks have no member/non-member labels and no ROC curve, so they do not derive from
:class:`~leakpro.attacks.mia_attacks.abstract_mia.AbstractMIA`. They plug into the scheduler
through the same three calls (``prepare_attack``, ``run_attack``, ``result.save``) and the same
``attack_id``.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel

from leakpro.attacks.attack_base import AbstractAttack
from leakpro.input_handler.extraction_handler import ExtractionHandler
from leakpro.reporting.extraction_result import ExtractionResult
from leakpro.utils.import_helper import Self
from leakpro.utils.save_load import fingerprint_model, hash_attack


class AbstractExtractionAttack(AbstractAttack):
    """Shared state for extraction attacks.

    Subclasses set ``AttackConfig`` (a pydantic model) and implement ``description``,
    ``prepare_attack`` and ``run_attack``.

    Args:
    ----
        handler: The extraction handler.
        configs: The attack's entry from ``audit.attack_list`` (without the ``attack`` key).

    """

    AttackConfig: type = BaseModel
    optuna_params = 0  # no hyperparameter search for extraction attacks

    # AbstractAttack.__init__ only creates an `attack_cache_folder` in the working directory, which
    # extraction does not use, so it is deliberately not called.
    def __init__(self: Self, handler: ExtractionHandler, configs: Optional[Dict[str, Any]]) -> None:  # noqa: D107
        self.handler = handler
        self.configs = self.AttackConfig(**(configs or {}))
        self.output_dir = handler.configs.audit.output_dir
        self.seed = int(handler.configs.audit.random_seed)
        self._hash_attack()

    def _hash_attack(self: Self) -> None:
        """Attack id from the config and a cheap fingerprint of the target (full hashing reads every weight)."""
        self.attack_id = hash_attack(self.configs.model_dump(), self.handler.target_model, model_hasher=fingerprint_model)

    @property
    def target_model(self: Self) -> Any:  # noqa: ANN401
        """The audited model."""
        return self.handler.target_model

    @abstractmethod
    def description(self: Self) -> dict:
        """Title, reference, summary and detailed description of the attack."""

    @abstractmethod
    def prepare_attack(self: Self) -> None:
        """Set up everything the attack needs before running."""

    @abstractmethod
    def run_attack(self: Self) -> ExtractionResult:
        """Run the attack."""
