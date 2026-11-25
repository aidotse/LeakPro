#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import copy

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.presets.gias import gias
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, TargetInfo
from leakpro.tests.gia import helpers
from leakpro.tests.gia.stand_ins import TinyClassifier, TinyGenerator


class TestGiasPreset:
    def test_two_stage_attack_produces_correct_shape(self):
        device = torch.device("cpu")
        model = TinyClassifier((3, 4, 4))
        images = torch.rand(1, 3, 4, 4)
        labels = torch.tensor([0])
        loss = nn.functional.cross_entropy(model(images), labels)
        gradients = torch.autograd.grad(loss, list(model.parameters()))
        obs = ClientObservation(gradients=[g.detach() for g in gradients])

        generator = TinyGenerator(8, (3, 4, 4))
        params_before = [p.detach().clone() for p in generator.parameters()]

        atk = gias(
            generator=generator, latent_dim=8, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
            device=device, num_classes=2,
            stage1_max_iter=2, stage2_max_iter=2,
        )
        state, ctx = run_attack(
            atk, model=model, observation=obs, input_shape=(1, 3, 4, 4), device=device,
            target=TargetInfo(value=labels),
        )
        assert state.input.shape == (1, 3, 4, 4)

        # Stage 2 (UnfrozenGANRepresentation) exposes the generator's own weights
        # as optimizable and fine-tunes them; stage 1 (GANRepresentation) never
        # touches them. If stage 2 silently reused GANRepresentation instead, the
        # generator's parameters would be unchanged after the attack even though
        # state.input's shape would look identical — so assert they actually moved.
        # The weights that move are the stage's private copy, carried in the state; the caller's generator is untouched.
        params_after = list(state.artifacts["generator_state"].values())
        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        for before, after in zip(params_before, generator.parameters()):
            assert torch.equal(before, after)


def _gias_run(stop=None, start=0, input_state=None):
    torch.manual_seed(3)
    model = TinyClassifier((3, 4, 4))
    labels = torch.tensor([0])
    loss = nn.functional.cross_entropy(model(torch.rand(1, 3, 4, 4)), labels)
    obs = ClientObservation(gradients=[g.detach() for g in torch.autograd.grad(loss, list(model.parameters()))])
    generator = TinyGenerator(8, (3, 4, 4))
    atk = gias(
        generator=generator, latent_dim=8, mean=[0.5] * 3, std=[0.5] * 3, device=torch.device("cpu"),
        num_classes=2, stage1_lr=0.1, stage1_max_iter=4, stage2_lr=0.5, stage2_max_iter=8,
    )
    state, _ = run_attack(
        atk, model=model, observation=obs, input_shape=(1, 3, 4, 4), device=torch.device("cpu"),
        target=TargetInfo(value=labels), start=start, stop=stop, input_state=input_state,
    )
    return state, generator


class TestGiasCarriesTheFineTunedGenerator:
    def test_carried_generator_weights_decode_the_returned_latent_to_the_returned_image(self):
        state, generator = _gias_run()

        carried = TinyGenerator(8, (3, 4, 4))
        carried.load_state_dict(state.artifacts["generator_state"])
        with torch.no_grad():
            decoded = (carried(state.artifacts["latent"]) + 1.0) / 2.0
        torch.testing.assert_close(decoded, state.input)
        # Fine-tuning moved the weights; the caller's generator is untouched.
        assert any(not torch.equal(a, b) for a, b in zip(carried.parameters(), generator.parameters()))

    def test_stage_two_alone_from_stage_one_output_equals_the_full_run(self):
        full, _ = _gias_run()
        stage_one, _ = _gias_run(stop=1)
        resumed, _ = _gias_run(start=1, input_state=stage_one)

        torch.testing.assert_close(resumed.input, full.input)
        for name, value in full.artifacts["generator_state"].items():
            torch.testing.assert_close(resumed.artifacts["generator_state"][name], value)


def test_gan_attack_leaves_the_callers_generator_untouched() -> None:
    generator = TinyGenerator(latent_dim=8, image_shape=helpers.IMAGE_SHAPE).train()
    before = copy.deepcopy(generator.state_dict())
    flags = [p.requires_grad for p in generator.parameters()]
    attack = gias(generator, 8, mean=helpers.CIFAR_MEAN, std=helpers.CIFAR_STD, device=helpers.DEVICE,
                  num_classes=helpers.NUM_CLASSES, stage1_max_iter=2, stage2_max_iter=2)
    model = helpers.tiny_bn_cnn().train()
    images, labels = helpers.client_batch()
    obs = helpers.observe(attack, model, images, labels)
    run_attack(attack, model, obs, tuple(images.shape), helpers.DEVICE, target=TargetInfo(value=labels))
    assert generator.training
    assert [p.requires_grad for p in generator.parameters()] == flags
    for k, v in generator.state_dict().items():
        torch.testing.assert_close(v, before[k])
