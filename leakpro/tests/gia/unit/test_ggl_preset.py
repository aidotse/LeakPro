#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch
from torch import nn

pytest.importorskip("cma")

from leakpro.attacks.gia_attacks.modular.presets.ggl import ggl
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, TargetInfo
from leakpro.tests.gia.stand_ins import TinyClassifier, TinyGenerator


class TestGglPreset:
    def test_cma_es_attack_produces_correct_shape(self):
        device = torch.device("cpu")
        model = TinyClassifier((3, 4, 4))
        model.train()
        model.to(device)
        images = torch.rand(1, 3, 4, 4)
        labels = torch.tensor([0])
        loss = nn.functional.cross_entropy(model(images), labels)
        gradients = torch.autograd.grad(loss, list(model.parameters()))
        obs = ClientObservation(gradients=[g.detach() for g in gradients])

        atk = ggl(
            generator=TinyGenerator(4, (3, 4, 4)), latent_dim=4, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
            device=device, num_classes=2,
            population_size=4, max_generations=2,
        )
        state, ctx = run_attack(
            atk, model=model, observation=obs, input_shape=(1, 3, 4, 4), device=device,
            target=TargetInfo(value=labels),
        )
        assert state.input.shape == (1, 3, 4, 4)
