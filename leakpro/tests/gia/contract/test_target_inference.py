#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""TargetInference contract: one label per sample, and the context's model is left as it was."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.interfaces import TargetInference
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.tests.gia.contract.conftest import make_ctx
from leakpro.tests.gia.contract.registry import build, cases

pytestmark = pytest.mark.parametrize(("cls", "factory"), cases(TargetInference))


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_infer_returns_one_label_per_sample_and_leaves_the_model_unchanged(cls, factory, mode) -> None:  # noqa: ANN001
    case = build(cls, factory)
    ctx = make_ctx(protocol=case.protocol)
    ctx.model.train(mode == "train")
    before = {k: v.clone() for k, v in ctx.model.state_dict().items()}

    target = case.component.infer(ctx)

    assert isinstance(target, TargetInfo)
    assert tuple(target.value.shape) == (ctx.input_shape[0],)
    assert ctx.model.training is (mode == "train")
    for name, value in ctx.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)
