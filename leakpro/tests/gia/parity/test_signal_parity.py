#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The victim's observable update equals v1's for every recorded training setting."""
import copy

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.tests.gia import helpers
from leakpro.tests.gia.parity import fixtures
from leakpro.tests.gia.parity.deltas import delta
from leakpro.tests.gia.parity.v1_equivalent import task


@pytest.mark.parametrize("case", [pytest.param(c, marks=delta(c)) for c in fixtures.cases("signal")])
def test_victim_signal_matches_v1(case: str) -> None:
    fx = fixtures.load(case)
    s = fx["settings"]
    victim = copy.deepcopy(fx["model"]).train(s["model_mode"] == "train")
    protocol = task({"compute_mode": s["compute_mode"], "model_mode": s["model_mode"], "epochs": s["epochs"],
                     "batch_size": s["batch_size"], "meta_learning_rate": s["meta_lr"]})
    obs = protocol.victim_signal(victim, helpers.image_modality().to_model_input(fx["images"]),
                                     TargetInfo(value=fx["labels"]))
    assert len(obs.gradients) == len(fx["gradients"])
    for i, (ours, ref) in enumerate(zip(obs.gradients, fx["gradients"])):
        torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-6, msg=f"parameter #{i}")
