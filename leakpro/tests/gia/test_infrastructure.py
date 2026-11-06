#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The GIA test infrastructure itself: determinism, helpers."""
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.tests.gia import helpers


def test_deterministic_algorithms_enabled() -> None:
    assert torch.are_deterministic_algorithms_enabled()


def test_client_batch_is_reproducible_with_distinct_labels() -> None:
    a_img, a_lab = helpers.client_batch(batch_size=4, seed=3)
    b_img, b_lab = helpers.client_batch(batch_size=4, seed=3)
    torch.testing.assert_close(a_img, b_img)
    assert torch.equal(a_lab, b_lab)
    assert len(set(a_lab.tolist())) == 4
    assert a_img.min() >= 0.0 and a_img.max() <= 1.0


def test_tiny_model_output_shape() -> None:
    assert helpers.tiny_bn_cnn()(torch.zeros(2, *helpers.IMAGE_SHAPE)).shape == (2, helpers.NUM_CLASSES)


def test_observe_does_not_mutate_the_callers_model() -> None:
    attack = presets.inverting_gradients(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE)
    model = helpers.tiny_bn_cnn().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    helpers.observe(attack, model, *helpers.client_batch())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], msg=key)


def test_observe_returns_a_nonzero_gradient_per_parameter() -> None:
    attack = presets.inverting_gradients(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE)
    model = helpers.tiny_bn_cnn().train()
    obs = helpers.observe(attack, model, *helpers.client_batch())
    assert len(obs.gradients) == len(list(model.parameters()))
    assert all(g.abs().sum() > 0 for g in obs.gradients)
    assert obs.bn_stats is not None and obs.batch_size == 2
