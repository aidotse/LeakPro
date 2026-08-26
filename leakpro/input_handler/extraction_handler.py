#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Runtime composition for extraction-specific user inputs."""

from __future__ import annotations

import inspect

from leakpro.input_handler.abstract_extraction_input_handler import AbstractExtractionInputHandler


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

    def __getattr__(self, name: str) -> object:
        """Lazily bind only methods declared by the extraction provider contract."""
        if name not in self.__dict__.get("_provider_methods", set()):
            raise AttributeError(name)
        return getattr(self._get_provider(), name)
