#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.core.loss_eval import evaluate_stage_losses
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossScope
from leakpro.attacks.gia_attacks.modular.core.state import Context, TargetInfo
from leakpro.tests.gia import helpers


def _ctx(attack, obs, shape):
    return Context(model=helpers.tiny_bn_cnn().train(), observation=obs, modality=attack.modality, task=attack.task,
                   input_shape=shape, device=helpers.DEVICE)


def test_gradient_matching_vanishes_at_the_client_batch() -> None:
    attack = presets.inverting_gradients(mean=helpers.CIFAR_MEAN, std=helpers.CIFAR_STD, device=helpers.DEVICE)
    images, labels = helpers.client_batch()
    obs = helpers.observe(attack, helpers.tiny_bn_cnn().train(), images, labels)
    ctx = _ctx(attack, obs, tuple(images.shape))
    ctx.session.begin_stage(ctx)
    out = evaluate_stage_losses(attack.stages[0].losses, ctx, images, TargetInfo(value=labels))
    assert out.terms["grad_match"].item() < 1e-5
    assert set(out.terms) == {"grad_match", "tv"}
    torch.testing.assert_close(out.total, sum(out.terms.values()))


def test_group_scope_evaluates_only_group_terms() -> None:
    attack = presets.yin(mean=helpers.CIFAR_MEAN, std=helpers.CIFAR_STD, device=helpers.DEVICE, num_seeds=2)
    x = torch.rand(2, 2, *helpers.IMAGE_SHAPE)
    ctx = _ctx(attack, None, (2, *helpers.IMAGE_SHAPE))
    out = evaluate_stage_losses(attack.stages[0].losses, ctx, x, TargetInfo(value=torch.tensor([0, 1])),
                                scope=LossScope.GROUP, full_recon=x)
    assert set(out.terms) == {"group"}
