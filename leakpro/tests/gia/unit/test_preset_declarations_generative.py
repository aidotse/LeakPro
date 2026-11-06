#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Every generative-prior preset states what the attacker knows."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.tests.gia import helpers, stand_ins

_GEN_KW = {"mean": [0.5] * 3, "std": [0.5] * 3, "device": helpers.DEVICE, "num_classes": 2}


def _generator() -> torch.nn.Module:
    return stand_ins.TinyGenerator(latent_dim=8, image_shape=helpers.IMAGE_SHAPE)


BUILDERS = {
    "gias": lambda: presets.gias(_generator(), 8, **_GEN_KW),
    "gifd": lambda: presets.gifd(stand_ins.tiny_layered_generator(), 8, **_GEN_KW),
    "ggl": lambda: presets.ggl(_generator(), 8, **_GEN_KW),
    "ggdm": lambda: presets.ggdm(stand_ins.ZeroUNet(), stand_ins.tiny_ddpm_scheduler(), **_GEN_KW),
    "gradinvdiff": lambda: presets.gradinvdiff(stand_ins.ZeroUNet(), stand_ins.tiny_ddpm_scheduler(), **_GEN_KW),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_preset_declares_a_threat_model(name: str) -> None:
    assert BUILDERS[name]().threat_model is not None, f"{name} leaves its threat model undeclared"
