#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import torch
from torch import nn

import pytest

from leakpro.attacks.gia_attacks.modular.components.representations.gifd import GIFDIntermediateRepresentation
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, State
from leakpro.tests.gia.stand_ins import tiny_layered_generator


def _ctx() -> Context:
    return Context(
        model=nn.Module(), observation=ClientObservation(), modality=None, task=None,
        input_shape=(1, 1, 8, 8), device=torch.device("cpu"), axes=AxisManager(),
    )


class TestGIFDIntermediateRepresentation:
    def test_bind_reads_features_from_state_artifacts(self):
        wrapper = tiny_layered_generator(latent_dim=4, channels=1)
        seed_features = torch.randn(1, 4, 1, 1)
        rep = GIFDIntermediateRepresentation(wrapper, start_layer=1, l1_radius=10.0)
        state = State(artifacts={"features": seed_features, "start_layer": 1})
        rep = rep.bind(state, _ctx())
        assert torch.equal(rep.parameters()[0]["params"][0].detach(), seed_features)

    def test_decode_runs_from_start_layer_onward(self):
        wrapper = tiny_layered_generator(latent_dim=4, channels=1)
        # start_layer=1 consumes the output of layer group 0 (8 channels), not
        # the raw latent (4 channels) — the brief's original draft used 4
        # channels here, which raises a channel-mismatch RuntimeError inside
        # ConvTranspose2d. Fixed to match LayeredGANWrapper's documented
        # forward_from contract (start_layer > 0 expects intermediate features).
        seed_features = torch.randn(1, 8, 1, 1)
        rep = GIFDIntermediateRepresentation(wrapper, start_layer=1, l1_radius=10.0)
        state = State(artifacts={"features": seed_features, "start_layer": 1})
        rep = rep.bind(state, _ctx())
        out = rep.decode()
        expected = (wrapper.forward_from(seed_features, start_layer=1) + 1.0) / 2.0  # Tanh range -> [0, 1]
        torch.testing.assert_close(out, expected)

    def test_project_keeps_deviation_within_l1_radius(self):
        # project() constrains the DEVIATION from the stage's initial (bind-time)
        # features, not the absolute feature magnitude — matching the legacy
        # per-stage shrinking radius. Simulate an optimization step moving the
        # features away from their anchor, then check the deviation is bounded.
        wrapper = tiny_layered_generator(latent_dim=4, channels=1)
        seed_features = torch.randn(1, 4, 1, 1)
        rep = GIFDIntermediateRepresentation(wrapper, start_layer=1, l1_radius=1.0)
        state = State(artifacts={"features": seed_features, "start_layer": 1})
        rep = rep.bind(state, _ctx())
        with torch.no_grad():
            rep.parameters()[0]["params"][0].add_(100.0)  # move far from the anchor
        rep.project(_ctx())
        deviation = rep.parameters()[0]["params"][0].detach() - rep._initial_features
        assert deviation.abs().sum().item() <= 1.0 + 1e-4

    def test_project_never_moves_features_already_at_their_initial_value(self):
        # Anchor-invariance check: if the features still equal their initial
        # (bind-time) value, the deviation is zero, so project() must leave
        # them completely unchanged regardless of l1_radius — even a radius of
        # 0.0 must not perturb them, since project_l1_ball(zeros, radius) is a
        # no-op for any radius >= 0.
        wrapper = tiny_layered_generator(latent_dim=4, channels=1)
        seed_features = torch.randn(1, 4, 1, 1) * 100  # large, but untouched since bind()
        rep = GIFDIntermediateRepresentation(wrapper, start_layer=1, l1_radius=0.0)
        state = State(artifacts={"features": seed_features, "start_layer": 1})
        rep = rep.bind(state, _ctx())
        rep.project(_ctx())
        assert torch.equal(rep.parameters()[0]["params"][0].detach(), seed_features)


class TestGifdHandoff:
    def test_stage_one_starts_from_the_first_layer_applied_to_the_latent(self):
        wrapper = tiny_layered_generator(latent_dim=4, channels=1)
        latent = torch.randn(1, 4)
        rep = GIFDIntermediateRepresentation(wrapper, start_layer=1, l1_radius=10.0)
        rep = rep.bind(State(artifacts={"latent": latent}), _ctx())
        expected = wrapper.apply_layer(latent.unsqueeze(-1).unsqueeze(-1), layer_idx=0)
        torch.testing.assert_close(rep.artifacts()["features"], expected)
        assert rep.artifacts()["start_layer"] == 1

    def test_later_stage_applies_exactly_one_layer_group_to_the_previous_features(self):
        wrapper = tiny_layered_generator(latent_dim=4, channels=1)
        features = torch.randn(1, 8, 4, 4)  # the output of layer group 0
        rep = GIFDIntermediateRepresentation(wrapper, start_layer=2, l1_radius=10.0)
        rep = rep.bind(State(artifacts={"features": features, "start_layer": 1}), _ctx())
        expected = wrapper.apply_layer(features, layer_idx=1)
        torch.testing.assert_close(rep.artifacts()["features"], expected)

    def test_unrelated_predecessor_is_rejected(self):
        wrapper = tiny_layered_generator(latent_dim=4, channels=1)
        rep = GIFDIntermediateRepresentation(wrapper, start_layer=2, l1_radius=10.0)
        with pytest.raises(ValueError, match="start_layer' 1"):
            rep.bind(State(artifacts={"latent": torch.randn(1, 4)}), _ctx())
