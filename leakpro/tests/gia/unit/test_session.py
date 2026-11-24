#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import copy

import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.components.target_inference import GengInference
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.tests.gia import helpers

KW = {"mean": helpers.CIFAR_MEAN, "std": helpers.CIFAR_STD, "device": helpers.DEVICE}


def _run(attack, model=None, target=True):
    model = model if model is not None else helpers.tiny_bn_cnn().train()
    images, labels = helpers.client_batch()
    obs = helpers.observe(attack, model, images, labels)
    return run_attack(attack, model, obs, tuple(images.shape), helpers.DEVICE,
                      target=TargetInfo(value=labels) if target else None)


def test_components_keep_no_run_state() -> None:
    attack = presets.dlg(**KW, num_classes=helpers.NUM_CLASSES, max_iter=2)
    stage = attack.stages[0]
    before = (dict(vars(stage.rep)), dict(vars(stage.target_source)))
    _run(attack, target=False)
    assert dict(vars(stage.rep)) == before[0]
    assert dict(vars(stage.target_source)) == before[1]


def test_bn_loss_holds_no_hooks_after_a_run() -> None:
    attack = presets.huang(**KW, max_iter=1)
    _run(attack)
    assert attack.stages[0].losses["bn_stats"][0].hooks == []


def test_geng_inference_keeps_no_state() -> None:
    inference = GengInference(num_dummy_samples=4)
    state_before = dict(vars(inference))
    attack = presets.inverting_gradients(**KW, max_iter=1)
    attack.target_inference = inference
    _run(attack, target=False)
    assert vars(inference) == state_before


def test_dimitrov_inference_restores_the_model_weights_and_mode() -> None:
    from leakpro.attacks.gia_attacks.modular.components.target_inference import DimitrovInference
    from leakpro.attacks.gia_attacks.modular.core.state import Context

    attack = presets.dimitrov(**KW, num_epochs=1, max_iter=1)
    model = helpers.tiny_bn_cnn().train()
    images, labels = helpers.client_batch()
    obs = helpers.observe(attack, model, images, labels)
    before = copy.deepcopy(model.state_dict())
    ctx = Context(model=model, observation=obs, modality=attack.modality, task=attack.task,
                  input_shape=tuple(images.shape), device=helpers.DEVICE)
    DimitrovInference(epochs=1, batch_size=2, num_dummy_samples=4).infer(ctx)
    assert model.training
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], msg=k)
