#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: AGPL-3.0-only
#
"""Tests of the YOLO example, kept next to the AGPL code they load (run them with ``pytest examples/gia/coco_yolo_1_image/tests``)."""
from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import torch

# Deterministic cuBLAS kernels need this set before CUDA initialises; harmless on CPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(autouse=True)
def _deterministic() -> Iterator[None]:
    """Seed the global RNG and force deterministic kernels, as the library's GIA tests do."""
    torch.manual_seed(0)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)
