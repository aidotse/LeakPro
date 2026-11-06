#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Shared builders for the contract tests: a standard run context and a snapshot of an object's state."""
from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor, nn

from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import ClientProtocol, Modality
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD
from leakpro.attacks.gia_attacks.modular.core.simulate import simulate_client
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.tests.gia import helpers

SEEDS = 2
"""Size of the seed axis the contracts bind with when they exercise an axis prefix."""


def seed_axes(size: int = SEEDS) -> AxisManager:
    """A single seed axis."""
    return AxisManager(Axis("seed", size))


def epoch_axes(size: int = SEEDS) -> AxisManager:
    """A single epoch axis."""
    return AxisManager(Axis("epoch", size))


def make_ctx(
    *,
    axes: AxisManager | None = None,
    model: nn.Module | None = None,
    modality: Modality | None = None,
    task: Task | None = None,
    protocol: ClientProtocol | None = None,
    input_shape: tuple[int, ...] | None = None,
    observation: ClientObservation | None = None,
) -> Context:
    """The standard contract context: the tiny BN CNN, CIFAR normalization, cross-entropy under FedSGD.

    Without an ``observation`` the client batch of ``helpers.client_batch`` is observed through
    ``simulate_client`` (epoch-prefixed when ``axes`` has an epoch axis), so every observation field is set.
    """
    axes = axes if axes is not None else AxisManager()
    model = model if model is not None else helpers.tiny_bn_cnn().train()
    modality = modality if modality is not None else helpers.image_modality()
    task = task if task is not None else Task(CrossEntropy(), protocol if protocol is not None else FedSGD())
    images, labels = helpers.client_batch()
    input_shape = input_shape if input_shape is not None else tuple(images.shape)
    if observation is None:
        epochs = axes.get_size("epoch") if axes.get_dim("epoch") is not None else None
        data = images.expand(epochs, *images.shape).clone() if epochs is not None else images
        observation = simulate_client(model, data, TargetInfo(value=labels), task=task, modality=modality,
                                      input_shape=input_shape if epochs is not None else None)
    return Context(model=model, observation=observation, modality=modality, task=task,
                   input_shape=input_shape, device=helpers.DEVICE, axes=axes)


def client_target() -> TargetInfo:
    """The hard labels of the standard client batch."""
    return TargetInfo(value=helpers.client_batch()[1])


def _fingerprint(value: Any) -> Any:  # noqa: ANN401 -- any attribute value
    """What must stay equal for ``value`` to count as unchanged (identity plus observable content)."""
    if isinstance(value, Tensor):
        return ("tensor", id(value), value.detach().clone(), value.requires_grad)
    if isinstance(value, nn.Module):
        return ("module", id(value), {k: v.detach().clone() for k, v in value.state_dict().items()},
                [m.training for m in value.modules()], [p.requires_grad for p in value.parameters()])
    if isinstance(value, (list, tuple)):
        return ("seq", id(value), [id(v) for v in value])
    if isinstance(value, dict):
        return ("dict", id(value), {k: id(v) for k, v in value.items()})
    return ("value", id(value))


def snapshot(obj: object) -> dict[str, Any]:
    """Fingerprints of every attribute of ``obj`` (``vars`` or dataclass fields)."""
    if dataclasses.is_dataclass(obj) and not hasattr(obj, "__dict__"):
        attrs = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    else:
        attrs = dict(vars(obj))
    return {k: _fingerprint(v) for k, v in attrs.items()}


def _same(a: Any, b: Any) -> bool:  # noqa: ANN401
    if isinstance(a, Tensor):
        return isinstance(b, Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def assert_unchanged(obj: object, before: dict[str, Any]) -> None:
    """Fail naming each attribute of ``obj`` whose fingerprint differs from ``before``."""
    after = snapshot(obj)
    changed = sorted(k for k in before.keys() | after.keys() if not _same(before.get(k), after.get(k)))
    assert not changed, f"{type(obj).__name__} configuration changed: {changed}"
