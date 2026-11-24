#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""TargetInfo DTO: device movement, cloning, serialization, and generic field ops.

These cover the contract that makes TargetInfo an *open* DTO — arbitrary fields
(boxes, tokens, …) must survive every transformation the pipeline applies, and the
whole object must round-trip through torch.save/load for the HPO worker hand-off.
"""
from __future__ import annotations

import copy
import pickle

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.core.io import load_state, save_state
from leakpro.attacks.gia_attacks.modular.core.state import State, TargetInfo


class TestDynamicFields:
    def test_arbitrary_fields_are_stored(self):
        t = TargetInfo(boxes=torch.zeros(2, 4), tokens=torch.ones(2, 8, dtype=torch.long))
        assert t.boxes.shape == (2, 4)
        assert t.tokens.dtype == torch.long

    def test_unset_field_is_none(self):
        assert TargetInfo(value=torch.zeros(1)).soft is None

    def test_axis_ndim_defaults_to_zero(self):
        assert TargetInfo(value=torch.zeros(1)).axis_ndim == 0

    def test_missing_dunder_raises_attribute_error(self):
        """Dunders must miss properly — copy/pickle probe the instance for them."""
        with pytest.raises(AttributeError):
            TargetInfo().__deepcopy__  # noqa: B018


class TestTransformsPreserveAllFields:
    @staticmethod
    def _target():
        return TargetInfo(
            value=torch.tensor([1, 2]),
            soft=torch.randn(2, 10),
            boxes=torch.randn(2, 4),
            meta="not-a-tensor",
            axis_ndim=0,
        )

    @pytest.mark.parametrize("op", ["clone", "detach", "to"])
    def test_extra_fields_survive(self, op):
        t = self._target()
        out = t.to("cpu") if op == "to" else getattr(t, op)()
        assert out.boxes.shape == (2, 4)
        assert out.meta == "not-a-tensor"

    def test_clone_does_not_alias(self):
        t = self._target()
        c = t.clone()
        c.boxes += 1.0
        assert not torch.allclose(c.boxes, t.boxes)

    def test_detach_clears_grad(self):
        t = TargetInfo(soft=torch.randn(2, 3, requires_grad=True))
        assert t.detach().soft.requires_grad is False

    def test_map_skips_non_tensors(self):
        t = self._target()
        out = t.map(lambda x: x * 0)
        assert out.meta == "not-a-tensor"
        assert torch.all(out.soft == 0)


class TestIndexAxis:
    def test_int_index_removes_dim_and_decrements_axis_ndim(self):
        t = TargetInfo(value=torch.zeros(4, 2, dtype=torch.long), soft=torch.randn(4, 2, 10), axis_ndim=1)
        out = t.index_axis(0, 2)
        assert out.value.shape == (2,)
        assert out.soft.shape == (2, 10)
        assert out.axis_ndim == 0

    def test_tensor_index_keeps_dim_and_axis_ndim(self):
        t = TargetInfo(value=torch.arange(4), soft=torch.randn(4, 10), axis_ndim=0)
        out = t.index_axis(0, torch.tensor([0, 2]))
        assert out.value.shape == (2,)
        assert out.soft.shape == (2, 10)
        assert out.axis_ndim == 0

    def test_extra_fields_are_sliced_too(self):
        """The whole point of index_axis: a new field is carried, not dropped."""
        t = TargetInfo(value=torch.zeros(3, 2, dtype=torch.long), boxes=torch.randn(3, 2, 4), axis_ndim=1)
        out = t.index_axis(0, 1)
        assert out.boxes.shape == (2, 4)

    def test_axis_ndim_never_goes_negative(self):
        t = TargetInfo(value=torch.zeros(2, 3, dtype=torch.long), axis_ndim=0)
        assert t.index_axis(0, 0).axis_ndim == 0


class TestSerialization:
    """Regression: __getattr__ returning None for dunders made TargetInfo
    unpicklable, which silently broke save_state/load_state entirely."""

    def test_deepcopy(self):
        t = TargetInfo(value=torch.zeros(3), boxes=torch.randn(1, 4))
        assert copy.deepcopy(t).boxes.shape == (1, 4)

    def test_pickle_roundtrip(self):
        t = TargetInfo(value=torch.tensor([7]), axis_ndim=2)
        out = pickle.loads(pickle.dumps(t))
        assert out.value.item() == 7
        assert out.axis_ndim == 2

    def test_save_load_state_roundtrip(self, tmp_path):
        target = TargetInfo(
            value=torch.tensor([1, 2]),
            soft=torch.randn(2, 10),
            boxes=torch.randn(2, 4),
            axis_ndim=1,
        )
        state = State(input=torch.randn(2, 3, 4, 4), target=target, loss=0.5, iteration=7)
        path = tmp_path / "state.pt"
        save_state(state, path)
        restored = load_state(path)

        assert torch.equal(restored.target.value, target.value)
        assert torch.allclose(restored.target.soft, target.soft)
        assert torch.allclose(restored.target.boxes, target.boxes)  # arbitrary field survives
        assert restored.target.axis_ndim == 1
        assert restored.iteration == 7

    def test_load_refuses_pickled_objects(self, tmp_path):
        """Checkpoints load with weights_only=True: a file carrying an arbitrary object is rejected, not unpickled."""
        path = tmp_path / "untrusted.pt"
        torch.save({"input": torch.zeros(1), "target": None, "artifacts": {"x": _Payload()}, "metrics": {},
                    "loss": 1.0, "iteration": 0, "metadata_seed": 0}, path)
        with pytest.raises(pickle.UnpicklingError):
            load_state(path)


class _Payload:
    """Stands in for an arbitrary object an untrusted checkpoint could carry."""
