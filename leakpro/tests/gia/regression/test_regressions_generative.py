#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Regression tests for bugs that were fixed in the generative priors (GAN, GIFD, diffusion, CMA-ES).

Each test asserts the correct behaviour through the public entry points (presets, run_attack, components).
"""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.interfaces import LossTerm
from leakpro.attacks.gia_attacks.modular.core.state import Context, State, TargetInfo
from leakpro.tests.gia import helpers, stand_ins
from leakpro.tests.gia.regression.test_regressions import KW, _run


def test_gan_reconstruction_is_returned_in_data_space() -> None:
    generator = stand_ins.TinyGenerator(latent_dim=8, image_shape=helpers.IMAGE_SHAPE)
    attack = presets.gias(generator, 8, **KW, num_classes=helpers.NUM_CLASSES, stage1_max_iter=2, stage2_max_iter=1)
    state, _ = _run(attack)
    assert state.input.min() >= 0.0 and state.input.max() <= 1.0


def test_diffusion_reconstruction_is_returned_in_data_space() -> None:
    pytest.importorskip("diffusers")
    attack = presets.ggdm(stand_ins.ZeroUNet(), stand_ins.tiny_ddpm_scheduler(), **KW,
                          num_classes=helpers.NUM_CLASSES, num_steps=2)
    state, _ = _run(attack)
    assert state.input.min() >= 0.0 and state.input.max() <= 1.0


class _ModelInputSpy(LossTerm):
    """Records whether each evaluation came through the shared evaluator (the only path that sets model_input)."""

    def __init__(self) -> None:
        self.saw_model_input: list[bool] = []
        self.dummy_requires_grad: list[bool] = []

    def compute(self, inputs):
        self.saw_model_input.append(inputs.model_input is not None)
        if inputs.dummy_obs is not None:
            self.dummy_requires_grad.append(inputs.dummy_obs.gradients[0].requires_grad)
        return inputs.recon.sum() * 0.0


def test_cma_es_scores_through_the_shared_evaluator_without_a_second_order_graph() -> None:
    pytest.importorskip("cma")
    generator = stand_ins.TinyGenerator(latent_dim=4, image_shape=helpers.IMAGE_SHAPE)
    attack = presets.ggl(generator, 4, **KW, num_classes=helpers.NUM_CLASSES, population_size=4, max_generations=1)
    spy = _ModelInputSpy()
    attack.stages[0].losses["spy"] = (spy, 1.0)
    _run(attack)
    assert spy.saw_model_input and all(spy.saw_model_input)
    assert not any(spy.dummy_requires_grad)


def test_diffusion_guidance_scores_through_the_shared_evaluator() -> None:
    pytest.importorskip("diffusers")
    attack = presets.ggdm(stand_ins.ZeroUNet(), stand_ins.tiny_ddpm_scheduler(), **KW,
                          num_classes=helpers.NUM_CLASSES, num_steps=2)
    spy = _ModelInputSpy()
    attack.stages[0].losses["spy"] = (spy, 1.0)
    _run(attack)
    assert spy.saw_model_input and all(spy.saw_model_input)


def test_gias_stage_two_starts_from_stage_one_best_latent() -> None:
    generator = stand_ins.TinyGenerator(latent_dim=8, image_shape=helpers.IMAGE_SHAPE)
    attack = presets.gias(generator, 8, **KW, num_classes=helpers.NUM_CLASSES, stage1_max_iter=6, stage2_max_iter=1)
    model = helpers.tiny_bn_cnn().train()
    images, labels = helpers.client_batch()
    obs = helpers.observe(attack, model, images, labels)
    recorder = helpers.LossRecorder()
    state, ctx = run_attack(attack, model, obs, tuple(images.shape), helpers.DEVICE, target=TargetInfo(value=labels),
                            callbacks=[recorder], stop=1)
    assert state.loss == pytest.approx(min(recorder.losses), rel=1e-5)
    stage_one_best_latent = state.artifacts["latent"]
    rep = attack.stages[1].rep.bind(state, ctx)
    torch.testing.assert_close(rep.artifacts()["latent"], stage_one_best_latent)


def test_gan_representation_with_a_seed_axis_decodes_with_the_axis_prefix() -> None:
    from leakpro.attacks.gia_attacks.modular.components.representations.gan import GANRepresentation
    from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
    generator = stand_ins.TinyGenerator(latent_dim=8, image_shape=helpers.IMAGE_SHAPE)
    ctx = Context(model=helpers.tiny_bn_cnn(), observation=None, modality=helpers.image_modality(), task=None,
                  input_shape=(2, *helpers.IMAGE_SHAPE), device=helpers.DEVICE, axes=AxisManager(Axis("seed", 3)))
    rep = GANRepresentation(generator, latent_dim=8).bind(State(), ctx)
    assert rep.decode().shape == (3, 2, *helpers.IMAGE_SHAPE)
    assert rep.artifacts()["latent"].shape == (3, 2, 8)


def test_gifd_every_stage_returns_data_space() -> None:
    wrapper = stand_ins.tiny_layered_generator()
    attack = presets.gifd(wrapper, 8, **KW, num_classes=helpers.NUM_CLASSES, stage_max_iter=1)
    model = stand_ins.TinyClassifier(helpers.IMAGE_SHAPE, num_classes=helpers.NUM_CLASSES)
    images, labels = helpers.client_batch(batch_size=1)
    obs = helpers.observe(attack, model, images, labels)
    for stop in range(1, len(attack.stages) + 1):
        state, _ = run_attack(attack, model, obs, tuple(images.shape), helpers.DEVICE, target=TargetInfo(value=labels),
                              stop=stop)
        assert state.input.min() >= 0.0 and state.input.max() <= 1.0, f"stage {stop - 1}"


def test_gifd_attack_runs_without_a_caller_supplied_callback() -> None:
    wrapper = stand_ins.tiny_layered_generator()
    attack = presets.gifd(wrapper, 8, **KW, num_classes=helpers.NUM_CLASSES, stage_max_iter=2)
    model = stand_ins.TinyClassifier(helpers.IMAGE_SHAPE, num_classes=helpers.NUM_CLASSES)
    images, labels = helpers.client_batch(batch_size=1)
    obs = helpers.observe(attack, model, images, labels)
    state, ctx = run_attack(attack, model, obs, tuple(images.shape), helpers.DEVICE, target=TargetInfo(value=labels),
                            stop=1)
    rep = attack.stages[1].rep.bind(state, ctx)
    expected = wrapper.apply_layer(state.artifacts["latent"].reshape(-1, 8, 1, 1), 0).reshape(rep.artifacts()["features"].shape)
    torch.testing.assert_close(rep.artifacts()["features"], expected)
    full, _ = run_attack(attack, model, obs, tuple(images.shape), helpers.DEVICE, target=TargetInfo(value=labels))
    assert full.input.shape == images.shape
