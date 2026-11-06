#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.representations.gan import (
    GANRepresentation,
    UnfrozenGANRepresentation,
)
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, State
from leakpro.tests.gia.stand_ins import TinyGenerator


def _ctx(batch_size: int = 1) -> Context:
    return Context(
        model=nn.Module(), observation=ClientObservation(), modality=None, task=None,
        input_shape=(batch_size, 1, 4, 4), device=torch.device("cpu"), axes=AxisManager(),
    )


class TestGANRepresentation:
    def test_decode_maps_tanh_output_into_data_space(self):
        generator = TinyGenerator(4, (1, 4, 4))
        rep = GANRepresentation(generator, latent_dim=4)
        rep = rep.bind(State(), _ctx())
        out = rep.decode()
        assert out.shape == (1, 1, 4, 4)
        torch.testing.assert_close(out, (generator(rep.artifacts()["latent"]) + 1.0) / 2.0)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_output_range_sets_the_linear_map_to_data_space(self):
        generator = TinyGenerator(4, (1, 4, 4))
        rep = GANRepresentation(generator, latent_dim=4, output_range=(-1.0, 3.0)).bind(State(), _ctx())
        torch.testing.assert_close(rep.decode(), (generator(rep.artifacts()["latent"]) + 1.0) / 4.0)

    def test_output_range_must_be_increasing(self):
        import pytest

        with pytest.raises(ValueError, match="output_range"):
            GANRepresentation(TinyGenerator(4, (1, 4, 4)), latent_dim=4, output_range=(1.0, -1.0))

    def test_parameters_returns_latent_as_sole_leaf(self):
        rep = GANRepresentation(TinyGenerator(4, (1, 4, 4)), latent_dim=4)
        rep = rep.bind(State(), _ctx())
        params = rep.parameters()
        assert len(params) == 1
        assert params[0]["params"][0].requires_grad
        assert params[0]["params"][0].shape == (1, 4)

    def test_artifacts_carries_latent_for_stage_handoff(self):
        rep = GANRepresentation(TinyGenerator(4, (1, 4, 4)), latent_dim=4)
        rep = rep.bind(State(), _ctx())
        artifacts = rep.artifacts()
        assert artifacts["latent"].shape == (1, 4)
        assert not artifacts["latent"].requires_grad

    def test_bind_resumes_latent_from_state_artifacts(self):
        rep = GANRepresentation(TinyGenerator(4, (1, 4, 4)), latent_dim=4)
        state = State(artifacts={"latent": torch.ones(1, 4)})
        rep = rep.bind(state, _ctx())
        assert torch.equal(rep.parameters()[0]["params"][0].detach(), torch.ones(1, 4))

    def test_bind_derives_batch_size_from_input_shape_not_axes(self):
        # Regression test: batch_size must come from ctx.input_shape[0] (the
        # canonical [N, C, H, W] convention), not from ctx.axes.shape (the
        # seed/epoch prefix, which is unrelated to the image batch size).
        rep = GANRepresentation(TinyGenerator(4, (1, 4, 4)), latent_dim=4)
        rep = rep.bind(State(), _ctx(batch_size=2))
        params = rep.parameters()
        assert params[0]["params"][0].shape == (2, 4)
        out = rep.decode()
        assert out.shape == (2, 1, 4, 4)

    def test_spherical_project_preserves_initial_radius(self):
        rep = GANRepresentation(TinyGenerator(4, (1, 4, 4)), latent_dim=4, spherical=True)
        rep = rep.bind(State(), _ctx())
        initial_norm = rep.parameters()[0]["params"][0].detach().norm(dim=1)
        with torch.no_grad():
            rep.parameters()[0]["params"][0].add_(5.0)  # perturb off the sphere
        rep.project(_ctx())
        new_norm = rep.parameters()[0]["params"][0].detach().norm(dim=1)
        assert torch.allclose(new_norm, initial_norm, atol=1e-4)


    def test_encode_is_not_supported(self):
        import pytest

        rep = GANRepresentation(TinyGenerator(4, (1, 4, 4)), latent_dim=4).bind(State(), _ctx())
        with pytest.raises(NotImplementedError, match="cannot be encoded"):
            rep.encode(torch.zeros(1, 1, 4, 4))


class TestUnfrozenGANRepresentation:
    def test_parameters_returns_generator_weights_not_latent(self):
        generator = TinyGenerator(4, (1, 4, 4))
        rep = UnfrozenGANRepresentation(generator, latent_dim=4)
        state = State(artifacts={"latent": torch.zeros(1, 4)})
        rep = rep.bind(state, _ctx())
        params = rep.parameters()
        param_tensors = [p for pg in params for p in pg["params"]]
        # The run fine-tunes its own copy of the generator, never the caller's module.
        assert rep.generator is not generator
        assert any(pt is p for pt in param_tensors for p in rep.generator.parameters())
        assert not any(pt is p for pt in param_tensors for p in generator.parameters())

    def test_decode_uses_fixed_latent_from_artifacts(self):
        generator = TinyGenerator(4, (1, 4, 4))
        rep = UnfrozenGANRepresentation(generator, latent_dim=4)
        state = State(artifacts={"latent": torch.zeros(1, 4)})
        rep = rep.bind(state, _ctx())
        out = rep.decode()
        assert out.shape == (1, 1, 4, 4)


class TestGeneratorCopiesAreNotShared:
    def test_frozen_and_unfrozen_bindings_in_one_run_do_not_share_a_generator(self):
        generator = TinyGenerator(4, (1, 4, 4))
        ctx = _ctx()
        state = State(artifacts={"latent": torch.zeros(1, 4)})
        unfrozen = UnfrozenGANRepresentation(generator, latent_dim=4).bind(state, ctx)
        frozen = GANRepresentation(generator, latent_dim=4).bind(state, ctx)

        assert frozen.generator is not unfrozen.generator
        assert all(p.requires_grad for p in unfrozen.generator.parameters())
        assert not any(p.requires_grad for p in frozen.generator.parameters())

    def test_bindings_start_from_the_carried_generator_state(self):
        generator = TinyGenerator(4, (1, 4, 4))
        tuned = TinyGenerator(4, (1, 4, 4))
        carried = {k: v + 1.0 for k, v in tuned.state_dict().items()}
        state = State(artifacts={"latent": torch.zeros(1, 4), "generator_state": carried})

        for rep in (GANRepresentation(generator, 4), UnfrozenGANRepresentation(generator, 4)):
            bound = rep.bind(state, _ctx())
            for name, value in bound.generator.state_dict().items():
                torch.testing.assert_close(value, carried[name])
            assert "generator_state" in bound.artifacts()


class TestGeneratorWeightsInDriverArtifacts:
    """Step events see the generator weights as views; the driver clones them only for the result it keeps."""

    @staticmethod
    def _run(return_best: bool):
        from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
        from leakpro.attacks.gia_attacks.modular.components.losses import L2GradientLoss
        from leakpro.attacks.gia_attacks.modular.components.representations.gan import UnfrozenGANRepresentation
        from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
        from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback
        from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
        from leakpro.tests.gia import helpers
        from leakpro.tests.gia.contract.conftest import client_target, make_ctx
        from leakpro.tests.gia.stand_ins import TinyGenerator

        seen = []

        class _Spy(Callback):
            def on_step_end(self, event):  # noqa: ANN001
                seen.append(event.artifacts)

        rep = helpers.BoundRecorder(UnfrozenGANRepresentation(TinyGenerator(8, helpers.IMAGE_SHAPE), 8))
        stage = OptimStage(losses={"grad": (L2GradientLoss(), 1.0)}, rep=rep,
                           driver=GradientDriver(lr=0.05, max_iter=3, return_best=return_best, schedule=None),
                           target_source=FixedTarget())
        ctx = make_ctx()
        state = State(target=client_target(), artifacts={"latent": torch.randn(helpers.client_batch()[0].shape[0], 8)})
        stage.run(state, ctx, callbacks=[_Spy()])
        return rep.bound.generator, seen, state.artifacts["generator_state"]

    def test_step_events_carry_views_of_the_live_weights(self):
        generator, seen, _ = self._run(return_best=False)
        live = generator.state_dict()
        assert len(seen) == 3
        for artifacts in seen:
            for name, value in artifacts["generator_state"].items():
                assert value.data_ptr() == live[name].data_ptr()

    def test_returned_weights_are_a_copy_of_the_final_weights(self):
        generator, _, returned = self._run(return_best=False)
        for name, value in generator.state_dict().items():
            assert returned[name].data_ptr() != value.data_ptr()
            torch.testing.assert_close(returned[name], value)
