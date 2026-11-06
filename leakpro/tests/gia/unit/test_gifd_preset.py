#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.presets.gifd import gifd
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, TargetInfo
from leakpro.tests.gia.stand_ins import TinyClassifier, tiny_layered_generator


class TestGifdPreset:
    def test_progressive_multi_stage_attack_produces_correct_shape(self):
        device = torch.device("cpu")
        wrapper = tiny_layered_generator()
        model = TinyClassifier((3, 8, 8))
        images = torch.rand(1, 3, 8, 8)
        labels = torch.tensor([0])
        loss = nn.functional.cross_entropy(model(images), labels)
        gradients = torch.autograd.grad(loss, list(model.parameters()))
        obs = ClientObservation(gradients=[g.detach() for g in gradients])

        atk = gifd(
            wrapped_generator=wrapper, latent_dim=8, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
            device=device, num_classes=2, stage_max_iter=2,
        )
        # num_layer_groups intermediate feature stages minus one — a stage with
        # start_layer == num_layer_groups would be a no-op identity pass-through
        # (LayeredGANWrapper.forward_from loops range(start_layer, num_layer_groups)),
        # so it is never built. See presets/gifd.py for the full rationale.
        assert len(atk.stages) == wrapper.num_layer_groups
        for stage in atk.stages[1:]:
            assert stage.rep.start_layer < wrapper.num_layer_groups
        state, ctx = run_attack(
            atk, model=model, observation=obs, input_shape=(1, 3, 8, 8), device=device,
            target=TargetInfo(value=labels),
        )
        assert state.input.shape == (1, 3, 8, 8)
