#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for AxisManager.

Pure computation — no model, no tensors, no I/O.
"""



from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager


# ── shape ─────────────────────────────────────────────────────────────────────

class TestAxisManagerShape:
    def test_empty_yields_empty_tuple(self):
        assert AxisManager().shape == ()

    def test_single_axis(self):
        assert AxisManager(Axis("seed", 4)).shape == (4,)

    def test_two_axes_in_order(self):
        am = AxisManager(Axis("epoch", 3), Axis("seed", 5))
        assert am.shape == (3, 5)

    def test_axis_order_is_preserved(self):
        """Reversing registration order reverses the shape tuple."""
        am1 = AxisManager(Axis("a", 2), Axis("b", 7))
        am2 = AxisManager(Axis("b", 7), Axis("a", 2))
        assert am1.shape == (2, 7)
        assert am2.shape == (7, 2)


# ── get_dim ───────────────────────────────────────────────────────────────────

class TestAxisManagerGetDim:
    def test_first_axis_is_dim_0(self):
        am = AxisManager(Axis("epoch", 2), Axis("seed", 4))
        assert am.get_dim("epoch") == 0

    def test_second_axis_is_dim_1(self):
        am = AxisManager(Axis("epoch", 2), Axis("seed", 4))
        assert am.get_dim("seed") == 1

    def test_absent_axis_returns_none(self):
        am = AxisManager(Axis("seed", 4))
        assert am.get_dim("missing") is None

    def test_empty_manager_returns_none(self):
        assert AxisManager().get_dim("seed") is None


# ── get_size ──────────────────────────────────────────────────────────────────

class TestAxisManagerGetSize:
    def test_returns_correct_size(self):
        am = AxisManager(Axis("epoch", 3), Axis("seed", 5))
        assert am.get_size("epoch") == 3
        assert am.get_size("seed") == 5

    def test_absent_axis_defaults_to_one(self):
        am = AxisManager(Axis("seed", 4))
        assert am.get_size("absent") == 1

    def test_empty_manager_defaults_to_one(self):
        assert AxisManager().get_size("seed") == 1


# ── equality ──────────────────────────────────────────────────────────────────

class TestAxisManagerEquality:
    def test_identical_managers_are_equal(self):
        a = AxisManager(Axis("seed", 3))
        b = AxisManager(Axis("seed", 3))
        assert a == b

    def test_different_sizes_not_equal(self):
        assert AxisManager(Axis("seed", 3)) != AxisManager(Axis("seed", 4))

    def test_different_names_not_equal(self):
        assert AxisManager(Axis("seed", 3)) != AxisManager(Axis("epoch", 3))

    def test_different_order_not_equal(self):
        """Axis order is significant — (a, b) != (b, a)."""
        ab = AxisManager(Axis("a", 2), Axis("b", 3))
        ba = AxisManager(Axis("b", 3), Axis("a", 2))
        assert ab != ba

    def test_empty_managers_are_equal(self):
        assert AxisManager() == AxisManager()

    def test_empty_vs_nonempty_not_equal(self):
        assert AxisManager() != AxisManager(Axis("seed", 1))

    def test_non_axis_manager_returns_not_implemented(self):
        result = AxisManager().__eq__("not an AxisManager")
        assert result is NotImplemented


# ── repr ──────────────────────────────────────────────────────────────────────

class TestAxisManagerRepr:
    def test_empty_repr(self):
        assert repr(AxisManager()) == "AxisManager()"

    def test_single_axis_repr(self):
        assert repr(AxisManager(Axis("seed", 4))) == "AxisManager(seed=4)"

    def test_multi_axis_repr(self):
        am = AxisManager(Axis("epoch", 2), Axis("seed", 4))
        assert repr(am) == "AxisManager(epoch=2, seed=4)"
