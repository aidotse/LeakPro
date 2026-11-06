#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for JointTarget — bind, current, parameters.

Uses a SimpleNamespace stub for Context so no real model is needed.
"""
import types

import pytest
import torch


from leakpro.attacks.gia_attacks.modular.components.targets import JointTarget
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.state import State, TargetInfo

DEVICE = torch.device("cpu")
BATCH = 2
NUM_CLASSES = 10


def _ctx(axes: AxisManager | None = None) -> types.SimpleNamespace:
    """Minimal context stub — only the fields JointTarget.bind() needs."""
    return types.SimpleNamespace(
        input_shape=(BATCH, 3, 8, 8),
        device=DEVICE,
        axes=axes or AxisManager(),
    )


# ── bind — tensor shape ───────────────────────────────────────────────────────

class TestJointTargetBind:
    def test_no_axes_allocates_2d(self):
        """Without axes, shape is (batch, num_classes)."""
        jt = JointTarget(NUM_CLASSES)
        jt = jt.bind(State(), _ctx())
        assert jt._soft_label.shape == (BATCH, NUM_CLASSES)

    def test_single_axis_prepends_dim(self):
        """A single 'seed' axis of size S → (S, batch, num_classes)."""
        S = 3
        jt = JointTarget(NUM_CLASSES)
        jt = jt.bind(State(), _ctx(AxisManager(Axis("seed", S))))
        assert jt._soft_label.shape == (S, BATCH, NUM_CLASSES)

    def test_two_axes_prepend_both_dims(self):
        """Two axes of sizes E, S → (E, S, batch, num_classes)."""
        E, S = 4, 2
        jt = JointTarget(NUM_CLASSES)
        jt = jt.bind(State(), _ctx(AxisManager(Axis("epoch", E), Axis("seed", S))))
        assert jt._soft_label.shape == (E, S, BATCH, NUM_CLASSES)

    def test_allocated_tensor_has_requires_grad(self):
        jt = JointTarget(NUM_CLASSES)
        jt = jt.bind(State(), _ctx())
        assert jt._soft_label.requires_grad

    def test_allocated_tensor_on_correct_device(self):
        jt = JointTarget(NUM_CLASSES)
        jt = jt.bind(State(), _ctx())
        assert jt._soft_label.device.type == DEVICE.type

    def test_each_bind_returns_a_fresh_copy(self):
        """Each bind returns a new bound copy with a fresh tensor; the configured source keeps none."""
        jt = JointTarget(NUM_CLASSES)
        first = jt.bind(State(), _ctx())
        second = jt.bind(State(), _ctx())
        assert second is not first and second._soft_label is not first._soft_label
        assert jt._soft_label is None


# ── current ───────────────────────────────────────────────────────────────────

class TestJointTargetCurrent:
    def setup_method(self):
        self.jt = JointTarget(NUM_CLASSES)
        self.jt = self.jt.bind(State(), _ctx())

    def test_current_has_soft_set(self):
        ti = self.jt.current()
        assert ti.soft is not None

    def test_current_soft_is_the_allocated_label(self):
        ti = self.jt.current()
        assert ti.soft is self.jt._soft_label

    def test_current_value_is_argmax_of_soft(self):
        """value must equal argmax(soft, dim=-1) — shape (batch,) without axes."""
        ti = self.jt.current()
        expected = self.jt._soft_label.argmax(dim=-1)
        assert torch.equal(ti.value, expected)

    def test_current_value_shape_matches_batch(self):
        ti = self.jt.current()
        assert ti.value.shape == (BATCH,)

    def test_current_with_axes_value_has_prefix_shape(self):
        """With seed axis S, value shape is (S, batch) — argmax over last dim."""
        S = 3
        jt = JointTarget(NUM_CLASSES)
        jt = jt.bind(State(), _ctx(AxisManager(Axis("seed", S))))
        ti = jt.current()
        assert ti.value.shape == (S, BATCH)


# ── parameters ────────────────────────────────────────────────────────────────

class TestJointTargetParameters:
    def test_returns_one_group(self):
        jt = JointTarget(NUM_CLASSES)
        jt = jt.bind(State(), _ctx())
        assert len(jt.parameters()) == 1

    def test_group_contains_soft_label(self):
        jt = JointTarget(NUM_CLASSES)
        jt = jt.bind(State(), _ctx())
        group = jt.parameters()[0]
        assert jt._soft_label in group["params"]

    def test_group_has_lr_scale(self):
        jt = JointTarget(NUM_CLASSES)
        jt = jt.bind(State(), _ctx())
        group = jt.parameters()[0]
        assert "lr_scale" in group

    def test_group_name_is_soft_labels(self):
        jt = JointTarget(NUM_CLASSES)
        jt = jt.bind(State(), _ctx())
        assert jt.parameters()[0]["name"] == "soft_labels"


# ── reduce ────────────────────────────────────────────────────────────────────

class TestJointTargetReduce:
    """reduce(target, dim, method) collapses the seed-prefixed soft labels."""

    def _make_seeded_target(self, seed_size: int) -> tuple[JointTarget, TargetInfo]:
        S = seed_size
        jt = JointTarget(NUM_CLASSES)
        # Allocate manually so we control the values
        soft = torch.randn(S, BATCH, NUM_CLASSES)
        jt._soft_label = soft
        target = TargetInfo(value=soft.argmax(dim=-1), soft=soft)
        return jt, target

    def test_mean_reduces_soft_along_seed_dim(self):
        """After reduce with method='mean', soft.shape drops the seed dimension."""
        S = 4
        jt, target = self._make_seeded_target(S)
        result = jt.reduce(target, dim=0, method="mean")
        assert result.soft.shape == (BATCH, NUM_CLASSES)

    def test_mean_soft_equals_arithmetic_mean(self):
        """result.soft must equal target.soft.mean(dim=0) exactly."""
        S = 3
        jt, target = self._make_seeded_target(S)
        result = jt.reduce(target, dim=0, method="mean")
        expected = target.soft.mean(dim=0)
        assert torch.allclose(result.soft, expected, atol=1e-6)

    def test_mean_value_is_argmax_of_averaged_soft(self):
        """result.value must be recomputed as argmax(result.soft, dim=-1)."""
        S = 5
        jt, target = self._make_seeded_target(S)
        result = jt.reduce(target, dim=0, method="mean")
        expected_value = result.soft.argmax(dim=-1)
        assert torch.equal(result.value, expected_value)

    def test_mean_value_shape_equals_batch(self):
        S = 2
        jt, target = self._make_seeded_target(S)
        result = jt.reduce(target, dim=0, method="mean")
        assert result.value.shape == (BATCH,)

    def test_mean_at_non_zero_dim(self):
        """Reduce works correctly when the seed is at a non-zero dimension (e.g. dim=1)."""
        E, S = 2, 3
        jt = JointTarget(NUM_CLASSES)
        soft = torch.randn(E, S, BATCH, NUM_CLASSES)
        target = TargetInfo(value=soft.argmax(dim=-1), soft=soft)
        result = jt.reduce(target, dim=1, method="mean")
        # dim=1 (seed) collapsed → shape [E, BATCH, NUM_CLASSES]
        assert result.soft.shape == (E, BATCH, NUM_CLASSES)
        assert torch.allclose(result.soft, soft.mean(dim=1), atol=1e-6)

    def test_unsupported_method_raises(self):
        S = 2
        jt, target = self._make_seeded_target(S)
        with pytest.raises(ValueError, match="unsupported method"):
            jt.reduce(target, dim=0, method="median")

    def test_none_soft_raises(self):
        jt = JointTarget(NUM_CLASSES)
        target = TargetInfo(value=torch.tensor([0]), soft=None)
        with pytest.raises(ValueError, match="target.soft"):
            jt.reduce(target, dim=0, method="mean")


# ── TargetSource base reduce ──────────────────────────────────────────────────

class TestTargetSourceBaseReduce:
    """The base TargetSource.reduce() is a safe no-op for targets that carry no axis
    prefix, and raises NotImplementedError for targets that actually span the axis.

    Whether a target spans the axis is read from ``TargetInfo.axis_ndim`` (set by
    whichever TargetSource allocated it) rather than sniffed from tensor rank — rank
    sniffing cannot distinguish an unprefixed soft target [N, C] from a seed-prefixed
    hard target [S, N].
    """

    def test_unprefixed_value_is_a_noop(self):
        """value [N] with axis_ndim=0 → returned unchanged."""
        from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
        ft = FixedTarget()
        target = TargetInfo(value=torch.tensor([3]))
        result = ft.reduce(target, dim=0, method="mean")
        assert result is target

    def test_unprefixed_soft_is_a_noop(self):
        """soft [N, C] with axis_ndim=0 → returned unchanged, despite ndim > 1.

        This is the case the old rank-based heuristic got wrong.
        """
        from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
        ft = FixedTarget()
        target = TargetInfo(value=None, soft=torch.zeros(4, 10))
        result = ft.reduce(target, dim=0, method="mean")
        assert result is target

    def test_prefixed_value_raises(self):
        """value [S, N] with axis_ndim=1 → NotImplementedError."""
        from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
        ft = FixedTarget()
        target = TargetInfo(value=torch.zeros(3, 1), axis_ndim=1)
        with pytest.raises(NotImplementedError):
            ft.reduce(target, dim=0, method="mean")

    def test_prefixed_soft_raises(self):
        """soft [S, N, C] with axis_ndim=1 → NotImplementedError."""
        from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
        ft = FixedTarget()
        target = TargetInfo(value=None, soft=torch.zeros(3, 1, 10), axis_ndim=1)
        with pytest.raises(NotImplementedError):
            ft.reduce(target, dim=0, method="mean")

    def test_none_tensors_do_not_span(self):
        """Both value=None and soft=None → no spans → returns target unchanged."""
        from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
        ft = FixedTarget()
        target = TargetInfo(value=None, soft=None)
        result = ft.reduce(target, dim=0, method="mean")
        assert result is target
