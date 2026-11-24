#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Shared fixtures and slow-test gating for the GIA test suite."""
from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import torch

# Deterministic cuBLAS kernels need this set before CUDA initialises; harmless on CPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")



def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked gia_slow unless --run-slow is given."""
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="gia_slow: pass --run-slow to run")
    for item in items:
        if "gia_slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _deterministic() -> Iterator[None]:
    """Seed the global RNG and force deterministic kernels for every GIA test."""
    torch.manual_seed(0)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)
