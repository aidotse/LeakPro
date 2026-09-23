#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Runtime composition for extraction-specific user inputs."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from functools import cached_property
from typing import Any

from torch import Tensor

from leakpro.attacks.extraction_attacks.protocols import SamplingAdapter
from leakpro.attacks.extraction_attacks.utils_generative import condition_hash, extraction_audit_hash
from leakpro.input_handler.abstract_extraction_input_handler import AbstractExtractionInputHandler
from leakpro.utils.save_load import hash_config


class ExtractionHandler:
    """Bind an extraction provider to the validated LeakPro config."""

    def __init__(self, configs: object, provider_class: type[AbstractExtractionInputHandler]) -> None:
        self.configs = configs
        self.provider_class = provider_class
        self._provider: AbstractExtractionInputHandler | None = None
        self._provider_methods = {
            name for name, _method in inspect.getmembers(AbstractExtractionInputHandler, predicate=inspect.isfunction)
        }

    def _get_provider(self) -> AbstractExtractionInputHandler:
        if self._provider is None:
            self._provider = self.provider_class()
            self._provider.configs = self.configs
        return self._provider

    @cached_property
    def _adapter(self) -> SamplingAdapter:
        return self._get_provider().get_diffusion_adapter()

    def get_diffusion_adapter(self) -> SamplingAdapter:
        """Reuse the target adapter within this audit."""
        return self._adapter

    @cached_property
    def _reference_images(self) -> Tensor | None:
        return self._get_provider().get_extraction_reference_images()

    def get_extraction_reference_images(self) -> Tensor | None:
        """Load reference images once per audit, including absent references."""
        return self._reference_images

    @cached_property
    def _audit_hash(self) -> str:
        return extraction_audit_hash(
            self.configs.target.hash, conditions=None, reference_images=self._reference_images,
        )

    def get_audit_hash(self, conditions: Sequence[Any] | None = None) -> str:
        """Combine shared target/reference identity with attack-specific conditions."""
        if conditions is None:
            return self._audit_hash
        return hash_config({"audit_hash": self._audit_hash, "conditions": [condition_hash(c) for c in conditions]})

    def __getattr__(self, name: str) -> object:
        """Lazily bind only methods declared by the extraction provider contract."""
        if name not in self.__dict__.get("_provider_methods", set()):
            raise AttributeError(name)
        return getattr(self._get_provider(), name)
