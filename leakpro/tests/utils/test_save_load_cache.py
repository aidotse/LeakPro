#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the pickle-free cache helpers in leakpro.utils.save_load.

Covers:
- save_arrays / load_arrays round-trip arrays, scalars, lists and CUDA-or-CPU tensors
- object arrays are refused at write time (they would need pickling to read back)
- a pickle payload planted at a cache path raises instead of executing (GHSA-rw6c-3h2x-wh83, finding 1)
- load_tensors reads a torch.save'd tensor / state dict and refuses a pickled callable
"""
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from leakpro.utils.save_load import load_arrays, load_tensors, save_arrays


def test_save_and_load_arrays_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    payload = {
        "floats": np.arange(6, dtype=np.float32).reshape(2, 3),
        "ints": [1, 2, 3],
        "flag": True,
        "tensor": torch.arange(4),
        "tensor_list": [torch.tensor(0.5), torch.tensor(1.5)],
    }

    save_arrays(str(path), payload)
    loaded = load_arrays(str(path))

    assert set(loaded) == set(payload)
    np.testing.assert_array_equal(loaded["floats"], payload["floats"])
    np.testing.assert_array_equal(loaded["ints"], np.array([1, 2, 3]))
    assert loaded["flag"].item() is True
    np.testing.assert_array_equal(loaded["tensor"], np.arange(4))
    np.testing.assert_array_equal(loaded["tensor_list"], np.array([0.5, 1.5]))


def test_save_arrays_refuses_object_arrays(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="pickle-free"):
        save_arrays(str(tmp_path / "bad.npz"), {"objects": np.array([{"a": 1}, None], dtype=object)})


class _Marker:
    """Pickle payload that touches a file when unpickled."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):  # noqa: ANN204
        return (Path(self.marker).touch, ())


def test_load_arrays_does_not_execute_planted_pickle(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    planted = tmp_path / "trajectory_train_data.npz"
    with open(planted, "wb") as handle:
        pickle.dump(_Marker(str(marker)), handle)

    with pytest.raises((ValueError, OSError)):
        load_arrays(str(planted))
    assert not marker.exists(), "planted pickle must not run under load_arrays"

    # An .npz whose member is an object array is refused as well (allow_pickle=False).
    objects = tmp_path / "objects.npz"
    np.savez(objects, payload=np.array([_Marker(str(marker))], dtype=object))
    with pytest.raises(ValueError):
        load_arrays(str(objects))
    assert not marker.exists()


def test_load_tensors_round_trip_and_rejects_callables(tmp_path: Path) -> None:
    state = {"w": torch.ones(2, 2), "b": torch.zeros(2)}
    torch.save(state, tmp_path / "model.pt")
    loaded = load_tensors(str(tmp_path / "model.pt"))
    assert torch.equal(loaded["w"], state["w"]) and torch.equal(loaded["b"], state["b"])

    marker = tmp_path / "executed"
    torch.save({"w": torch.ones(1), "evil": _Marker(str(marker))}, tmp_path / "planted.pt")
    with pytest.raises(pickle.UnpicklingError):
        load_tensors(str(tmp_path / "planted.pt"))
    assert not marker.exists(), "planted callable must not run under load_tensors"
