#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Regression tests for bugs that were fixed in the framework.

Each test asserts the correct behaviour through the public entry points (presets, run_attack, components).
Tests labelled "Guard" protect properties that were never broken but would be easy to break.
"""
from __future__ import annotations

import copy

import pytest
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.components.losses.regularization import BNStatsLoss
from leakpro.attacks.gia_attacks.modular.core.attack import Attack, run_attack
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, State, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.tests.gia import helpers

KW = {"mean": helpers.CIFAR_MEAN, "std": helpers.CIFAR_STD, "device": helpers.DEVICE}


def _run(
    attack: Attack,
    *,
    oracle_target: bool = True,
    model: torch.nn.Module | None = None,
    images: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    callbacks: list | None = None,
) -> tuple[State, Context]:
    model = model if model is not None else helpers.tiny_bn_cnn().train()
    if images is None:
        images, labels = helpers.client_batch()
    obs = helpers.observe(attack, model, images, labels)
    target = TargetInfo(value=labels) if oracle_target else None
    return run_attack(attack, model, obs, tuple(images.shape), helpers.DEVICE, target=target, callbacks=callbacks)


def _assert_state_dict_unchanged(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> None:
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], msg=key)


def test_victim_signal_does_not_mutate_model() -> None:
    attack = presets.inverting_gradients(**KW)
    model = helpers.tiny_bn_cnn().train()
    before = copy.deepcopy(model.state_dict())
    images, labels = helpers.client_batch()
    attack.task.victim_signal(model, attack.modality.to_model_input(images), TargetInfo(value=labels))
    _assert_state_dict_unchanged(model, before)


def test_run_attack_does_not_mutate_model() -> None:
    """Guard: the attack simulates on a shadow copy, never on the caller's weights."""
    attack = presets.huang(**KW, max_iter=2)
    model = helpers.tiny_bn_cnn().train()
    before = copy.deepcopy(model.state_dict())
    _run(attack, model=model)
    assert model.training
    _assert_state_dict_unchanged(model, before)


def test_returned_input_reproduces_the_best_recorded_loss() -> None:
    """Guard: with a fixed target, the returned input is the best evaluated iterate.

    This also shows that stage_objective reproduces the driver's loss, which the joint-target test relies on.
    """
    attack = presets.inverting_gradients(**KW, max_iter=6)
    recorder = helpers.LossRecorder()
    state, ctx = _run(attack, callbacks=[recorder])
    at_returned_input = helpers.stage_objective(attack.stages[0], ctx, state.input, state.target).item()
    assert at_returned_input == pytest.approx(min(recorder.losses), rel=1e-5)


def test_joint_target_comes_from_the_same_iterate_as_the_image() -> None:
    attack = presets.dlg(**KW, num_classes=helpers.NUM_CLASSES, max_iter=6)
    recorder = helpers.LossRecorder()
    state, ctx = _run(attack, oracle_target=False, callbacks=[recorder])
    # No torch.no_grad(): gradient matching differentiates the dummy loss w.r.t. the model weights.
    at_returned_pair = helpers.stage_objective(attack.stages[0], ctx, state.input, state.target).item()
    assert at_returned_pair == pytest.approx(min(recorder.losses), rel=1e-4)


def test_state_records_loss_and_iterations() -> None:
    recorder = helpers.LossRecorder()
    state, _ = _run(presets.inverting_gradients(**KW, max_iter=3), callbacks=[recorder])
    assert state.loss == pytest.approx(min(recorder.losses))
    assert state.iteration == 3


def test_proxy_bn_without_dataloader_raises() -> None:
    attack = presets.huang(**KW, max_iter=1)
    attack.stages[0].losses["bn_stats"] = (BNStatsLoss(strategy="proxy"), 1.0)
    with pytest.raises(ValueError, match="proxy"):
        _run(attack)


def test_label_inference_preserves_model_mode() -> None:
    from leakpro.attacks.gia_attacks.modular.components.target_inference import DimitrovInference

    attack = presets.dimitrov(**KW, num_epochs=1, max_iter=1)
    attack.target_inference = DimitrovInference(epochs=1, batch_size=2, num_dummy_samples=4)
    model = helpers.tiny_bn_cnn().train()
    _run(attack, model=model, oracle_target=False)
    assert model.training


def test_sanitize_leaves_the_callers_observation_intact() -> None:
    obs = ClientObservation(gradients=[torch.ones(1)], bn_stats=[(torch.zeros(1), torch.ones(1))])
    sanitized = ThreatModel().sanitize(obs)
    assert sanitized.bn_stats is None
    assert obs.bn_stats is not None


def test_attack_object_is_reusable_with_identical_results() -> None:
    """Guard: running the same Attack twice gives identical reconstructions."""
    attack = presets.inverting_gradients(**KW, max_iter=3)
    first, _ = _run(attack)
    second, _ = _run(attack)
    torch.testing.assert_close(first.input, second.input)
