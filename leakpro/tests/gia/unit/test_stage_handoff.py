#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for OptimStage.run() state propagation.

Uses _ConstantDriver — a stub that returns a fixed tensor — so no gradient
optimization runs. Every test is validating what the stage does WITH the driver's
output, not the optimization itself.
"""

import torch
import torch.nn as nn


from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget, JointTarget
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback, Driver
from leakpro.attacks.gia_attacks.modular.core.result import DriverResult
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import (
    ClientObservation, Context, State, TargetInfo,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

INPUT_SHAPE = (1, 3, 8, 8)
NUM_CLASSES = 10
DEVICE = torch.device("cpu")
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]
_FIXED_OUT = torch.ones(*INPUT_SHAPE) * 0.42


class _ConstantDriver(Driver):
    """Returns a fixed tensor — no optimization, no loss evaluation."""
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def run(self, stage, ctx, *, callbacks=(), stage_idx=0):
        return DriverResult(input=self._tensor, target=stage.target_source.current(), artifacts=stage.rep.artifacts(),
                            loss=0.0, iterations=0, slice_losses=None)


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, NUM_CLASSES)).eval()


def _make_ctx(axes: AxisManager | None = None) -> Context:
    model = _make_model()
    task = Task(CrossEntropy())
    imgs = torch.rand(INPUT_SHAPE)
    obs_raw = task.victim_signal(model, imgs, TargetInfo(value=torch.tensor([0])))
    obs = ClientObservation(gradients=[g.detach().clone() for g in obs_raw.gradients])
    return Context(
        model=model,
        observation=obs,
        modality=ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE),
        task=task,
        input_shape=INPUT_SHAPE,
        device=DEVICE,
        axes=axes or AxisManager(),
    )


def _fixed_stage(driver=None, target_source=None) -> OptimStage:
    return OptimStage(
        losses={},
        rep=PixelRepresentation(),
        driver=driver or _ConstantDriver(_FIXED_OUT),
        target_source=target_source or FixedTarget(),
    )


# ── state.input ───────────────────────────────────────────────────────────────

class TestStageInput:
    def test_state_input_is_set_to_driver_output(self):
        """After run(), state.input must equal what the driver returned."""
        ctx = _make_ctx()
        state = State(target=TargetInfo(value=torch.tensor([0])))
        out = _fixed_stage().run(state, ctx)
        assert torch.equal(out.input, _FIXED_OUT)

    def test_state_input_replaces_any_prior_input(self):
        """A pre-existing state.input is overwritten by the stage output."""
        ctx = _make_ctx()
        prior = torch.zeros(*INPUT_SHAPE)
        state = State(input=prior, target=TargetInfo(value=torch.tensor([0])))
        out = _fixed_stage().run(state, ctx)
        assert torch.equal(out.input, _FIXED_OUT)
        assert not torch.equal(out.input, prior)


# ── state.target detachment ───────────────────────────────────────────────────

class TestTargetDetachment:
    def test_fixed_target_value_is_detached_after_stage(self):
        """FixedTarget's value must be stored in state as a non-grad tensor."""
        ctx = _make_ctx()
        orig = torch.tensor([3])
        state = State(target=TargetInfo(value=orig))
        out = _fixed_stage().run(state, ctx)
        assert not out.target.value.requires_grad

    def test_fixed_target_value_is_a_clone(self):
        """The target stored in state must be a clone, not the original object."""
        ctx = _make_ctx()
        orig = torch.tensor([3])
        state = State(target=TargetInfo(value=orig))
        out = _fixed_stage().run(state, ctx)
        assert out.target.value is not orig
        assert torch.equal(out.target.value, orig)

    def test_joint_target_soft_is_detached_after_stage(self):
        """JointTarget's soft labels must be stored detached (no autograd) in state."""
        ctx = _make_ctx()
        stage = _fixed_stage(target_source=JointTarget(NUM_CLASSES))
        out = stage.run(State(), ctx)
        assert not out.target.soft.requires_grad

    def test_joint_target_soft_shape_matches_batch_and_classes(self):
        ctx = _make_ctx()
        stage = _fixed_stage(target_source=JointTarget(NUM_CLASSES))
        out = stage.run(State(), ctx)
        assert out.target.soft.shape == (INPUT_SHAPE[0], NUM_CLASSES)

    def test_joint_target_value_is_argmax_of_soft(self):
        ctx = _make_ctx()
        stage = _fixed_stage(target_source=JointTarget(NUM_CLASSES))
        out = stage.run(State(), ctx)
        expected = out.target.soft.argmax(dim=-1)
        assert torch.equal(out.target.value, expected)


# ── state.artifacts ───────────────────────────────────────────────────────────

class TestStageArtifacts:
    def test_pixels_artifact_is_set_by_pixel_representation(self):
        ctx = _make_ctx()
        state = State(target=TargetInfo(value=torch.tensor([0])))
        out = _fixed_stage().run(state, ctx)
        assert "pixels" in out.artifacts

    def test_artifacts_replace_prior_unrelated_keys(self):
        """state.artifacts is fully replaced by rep.artifacts() — unrelated prior keys vanish.

        PixelRepresentation only returns {"pixels": ...}, so a "custom_key" that was
        in the incoming state must not appear in the outgoing state.
        """
        ctx = _make_ctx()
        state = State(
            target=TargetInfo(value=torch.tensor([0])),
            artifacts={"custom_key": torch.tensor([99.0])},
        )
        out = _fixed_stage().run(state, ctx)
        assert "custom_key" not in out.artifacts
        assert "pixels" in out.artifacts


# ── callbacks ─────────────────────────────────────────────────────────────────

class _Recorder(Callback):
    def __init__(self):
        self.starts: list[int] = []
        self.ends: list[int] = []

    def on_stage_start(self, stage_idx, state, ctx):
        self.starts.append(stage_idx)

    def on_stage_end(self, stage_idx, state, ctx):
        self.ends.append(stage_idx)


class TestStageCallbacks:
    def test_on_stage_start_called_once_with_correct_idx(self):
        ctx = _make_ctx()
        state = State(target=TargetInfo(value=torch.tensor([0])))
        cb = _Recorder()
        _fixed_stage().run(state, ctx, stage_idx=7, callbacks=[cb])
        assert cb.starts == [7]

    def test_on_stage_end_called_once_with_correct_idx(self):
        ctx = _make_ctx()
        state = State(target=TargetInfo(value=torch.tensor([0])))
        cb = _Recorder()
        _fixed_stage().run(state, ctx, stage_idx=7, callbacks=[cb])
        assert cb.ends == [7]

    def test_start_fires_before_end(self):
        ctx = _make_ctx()
        state = State(target=TargetInfo(value=torch.tensor([0])))
        events: list[str] = []

        class _OrderCb(Callback):
            def on_stage_start(self, *_): events.append("start")
            def on_stage_end(self, *_): events.append("end")

        _fixed_stage().run(state, ctx, callbacks=[_OrderCb()])
        assert events == ["start", "end"]

    def test_multiple_callbacks_all_fired(self):
        ctx = _make_ctx()
        state = State(target=TargetInfo(value=torch.tensor([0])))
        cbs = [_Recorder(), _Recorder()]
        _fixed_stage().run(state, ctx, callbacks=cbs)
        for cb in cbs:
            assert len(cb.starts) == 1
            assert len(cb.ends) == 1


# ── cross-stage propagation ───────────────────────────────────────────────────

class TestStagePropagation:
    def _run_two_stages(self, target_source_0, target_source_1, initial_target=None):
        ctx = _make_ctx()
        state = State(target=initial_target)
        state = _fixed_stage(target_source=target_source_0).run(state, ctx, stage_idx=0)
        state = _fixed_stage(target_source=target_source_1).run(state, ctx, stage_idx=1)
        return state

    def test_fixed_target_passes_through_two_stages_unchanged(self):
        """Hard labels placed in state.target survive two FixedTarget stages unchanged."""
        orig = torch.tensor([5])
        out = self._run_two_stages(
            FixedTarget(), FixedTarget(),
            initial_target=TargetInfo(value=orig),
        )
        assert torch.equal(out.target.value, orig)

    def test_joint_soft_flows_from_stage0_to_stage1(self):
        """Soft labels produced by JointTarget in stage 0 are preserved in stage 1.

        Stage 1 uses FixedTarget, which simply reads whatever is in state.target.
        After stage 1 the soft must equal the clone that stage 0 wrote.
        """
        ctx = _make_ctx()
        state = State()

        stage0 = _fixed_stage(target_source=JointTarget(NUM_CLASSES))
        state = stage0.run(state, ctx, stage_idx=0)
        soft_after_stage0 = state.target.soft.clone()

        stage1 = _fixed_stage(target_source=FixedTarget())
        state = stage1.run(state, ctx, stage_idx=1)

        assert state.target.soft is not None
        assert torch.equal(state.target.soft, soft_after_stage0)

    def test_stage0_input_seeds_stage1_representation(self):
        """PixelRepresentation.bind() picks up state.input from the previous stage.

        We verify indirectly: when stage 1 uses a _ConstantDriver that returns the
        same fixed tensor, the final state.input is that tensor regardless of the
        seed from stage 0.  This confirms bind() was called and did not crash.
        """
        ctx = _make_ctx()
        out = self._run_two_stages(
            FixedTarget(), FixedTarget(),
            initial_target=TargetInfo(value=torch.tensor([0])),
        )
        assert torch.equal(out.input, _FIXED_OUT)
