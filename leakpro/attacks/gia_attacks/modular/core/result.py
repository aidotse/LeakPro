#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""What a driver returns: the reconstruction, its target and artifacts from one iterate, and its loss."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo


@dataclass(frozen=True)
class DriverResult:
    """The outcome of one driver run.

    ``input``, ``target`` and ``artifacts`` describe the same iterate (per slice of the parallel
    axis, when there is one), so a later stage resuming from the artifacts starts exactly where the
    returned reconstruction came from.

    Attributes:
        input: Data space, detached, ``[*axes, *input_shape]``.
        target: The target of the same iterate(s) as ``input``, detached.
        artifacts: Representation artifacts of the same iterate(s), cloned. Tensor artifacts carry
            the ``ctx.axes`` prefix.
        loss: Mean over slices of the PER_SLICE objective at the returned iterate(s).
        iterations: Optimizer steps, generations or reverse steps executed.
        slice_losses: ``[slice count]`` PER_SLICE loss per slice; None without a parallel axis.

    """

    input: Tensor
    target: TargetInfo | None
    artifacts: dict[str, Any]
    loss: float
    iterations: int
    slice_losses: Tensor | None


def clone_artifacts(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    """Detached clones of every tensor artifact (also those inside a dict artifact); other values are carried as is."""
    return {k: _clone(v) for k, v in artifacts.items()}


def _clone(value: Any) -> Any:  # noqa: ANN401 -- values of any type are cloned
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {k: _clone(v) for k, v in value.items()}
    return value


def merge_slice(dst: Tensor, src: Tensor, dim: int | None, index: int) -> None:
    """Copy slice ``index`` along ``dim`` of ``src`` into ``dst`` in place (all of it when ``dim`` is None)."""
    with torch.no_grad():
        if dim is None:
            dst.copy_(src)
        else:
            dst.select(dim, index).copy_(src.select(dim, index))


def _check_prefix(name: str, value: Tensor, prefix: tuple[int, ...]) -> None:
    if tuple(value.shape[: len(prefix)]) != tuple(prefix):
        raise ValueError(
            f"Artifact {name!r} has shape {tuple(value.shape)}, which does not carry the axis prefix "
            f"{tuple(prefix)}; per-slice results cannot be merged or selected."
        )


def merge_artifact_slice(
    dst: dict[str, Any], src: Mapping[str, Any], prefix: tuple[int, ...], dim: int, index: int,
) -> None:
    """Merge slice ``index`` along ``dim`` of every tensor artifact of ``src`` into ``dst`` in place.

    Raises:
        ValueError: Naming the artifact, if a tensor artifact does not carry the axis ``prefix``.

    """
    for name, value in src.items():
        if not isinstance(value, Tensor):
            dst[name] = value
            continue
        _check_prefix(name, value, prefix)
        merge_slice(dst[name], value.detach(), dim, index)


def select_artifacts(
    artifacts: Mapping[str, Any], dim: int, index: int, prefix: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Slice ``index`` along ``dim`` of every tensor artifact, removing that dimension.

    Raises:
        ValueError: Naming the artifact, if ``prefix`` is given and a tensor artifact does not carry it.

    """
    out: dict[str, Any] = {}
    for name, value in artifacts.items():
        if isinstance(value, Tensor):
            if prefix is not None:
                _check_prefix(name, value, prefix)
            out[name] = value.select(dim, index)
        else:
            out[name] = value
    return out
