#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch
from torch import nn

diffusers = pytest.importorskip("diffusers")

from leakpro.attacks.gia_attacks.modular.presets.ggdm import ggdm
from leakpro.attacks.gia_attacks.modular.presets.gradinvdiff import gradinvdiff
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, TargetInfo
from leakpro.tests.gia.stand_ins import TinyClassifier, ZeroUNet, tiny_ddpm_scheduler


def _obs_and_labels(model: nn.Module):
    images = torch.rand(1, 3, 4, 4)
    labels = torch.tensor([0])
    loss = nn.functional.cross_entropy(model(images), labels)
    gradients = torch.autograd.grad(loss, list(model.parameters()))
    return ClientObservation(gradients=[g.detach() for g in gradients]), labels


class TestGgdmPreset:
    def test_produces_correct_shape(self):
        device = torch.device("cpu")
        model = TinyClassifier((3, 4, 4))
        obs, labels = _obs_and_labels(model)
        atk = ggdm(
            unet=ZeroUNet(), scheduler=tiny_ddpm_scheduler(),
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], device=device, num_classes=2,
        )
        state, ctx = run_attack(
            atk, model=model, observation=obs, input_shape=(1, 3, 4, 4), device=device,
            target=TargetInfo(value=labels),
        )
        assert state.input.shape == (1, 3, 4, 4)


class TestGradInvDiffPreset:
    def test_produces_correct_shape(self):
        device = torch.device("cpu")
        model = TinyClassifier((3, 4, 4))
        obs, labels = _obs_and_labels(model)
        atk = gradinvdiff(
            unet=ZeroUNet(), scheduler=tiny_ddpm_scheduler(),
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], device=device, num_classes=2,
            inner_steps=1,
        )
        state, ctx = run_attack(
            atk, model=model, observation=obs, input_shape=(1, 3, 4, 4), device=device,
            target=TargetInfo(value=labels),
        )
        assert state.input.shape == (1, 3, 4, 4)


def test_all_five_presets_importable_from_package():
    from leakpro.attacks.gia_attacks.modular.presets import gias, gifd, ggl, ggdm, gradinvdiff  # noqa: F401
