#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Label inference recovers the same label multiset as v1 from the same observation."""
import copy

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.target_inference import (
    DimitrovInference,
    GengInference,
    IDLGInference,
)
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context
from leakpro.tests.gia import helpers
from leakpro.tests.gia.parity import fixtures
from leakpro.tests.gia.parity.deltas import delta
from leakpro.tests.gia.parity.v1_equivalent import task

INFERERS = {"idlg": IDLGInference, "geng": GengInference, "dimitrov": lambda: DimitrovInference(epochs=1, batch_size=4)}


@pytest.mark.parametrize("case", [pytest.param(c, marks=delta(c)) for c in fixtures.cases("labels")])
def test_label_inference_matches_v1(case: str) -> None:
    fx = fixtures.load(case)
    s = fx["settings"]
    protocol = task({"compute_mode": s["compute_mode"], "model_mode": "train", "epochs": 1, "batch_size": None,
                     "meta_learning_rate": 0.01})
    ctx = Context(model=copy.deepcopy(fx["model"]), observation=ClientObservation(gradients=fx["gradients"], batch_size=4),
                  modality=helpers.image_modality(), task=protocol, input_shape=tuple(fx["images"].shape),
                  device=helpers.DEVICE)
    torch.manual_seed(0)
    ours = INFERERS[s["method"]]().infer(ctx).value
    assert sorted(ours.tolist()) == sorted(fx["inferred"].tolist())
