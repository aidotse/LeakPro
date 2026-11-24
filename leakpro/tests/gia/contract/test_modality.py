#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Modality contract: ``strip_model_input`` inverts ``to_model_input``."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.interfaces import Modality
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(Modality))


def test_strip_inverts_to_model_input(cls, factory) -> None:  # noqa: ANN001
    case = build(cls, factory)
    roundtrip = case.component.strip_model_input(case.component.to_model_input(case.data))
    torch.testing.assert_close(roundtrip, case.data)
