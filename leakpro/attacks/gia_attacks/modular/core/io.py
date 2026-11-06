#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""State serialization for stage checkpointing and HPO worker hand-off."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from leakpro.attacks.gia_attacks.modular.core.state import RunMetadata, State, TargetInfo

PathLike = Union[str, Path]


def save_state(state: State, path: PathLike) -> None:
    """Saves the state's reconstruction, target, artifacts, metrics, loss, iteration and seed to a checkpoint file.

    The target is stored as a plain dict of its fields, so the file holds only tensors and builtin containers and
    ``load_state`` can read it with ``torch.load(weights_only=True)``.
    """
    torch.save({
        "input": state.input,
        "target": dict(vars(state.target)) if state.target is not None else None,
        "artifacts": state.artifacts,
        "metrics": state.metrics,
        "loss": state.loss,
        "iteration": state.iteration,
        "metadata_seed": state.metadata.seed,
    }, path)


def load_state(
    path: PathLike,
    map_location: Union[str, torch.device, None] = None,
) -> State:
    """Loads a state written by ``save_state``.

    The file is read with ``torch.load(weights_only=True)``: anything but tensors and builtin containers (for
    example a pickled object in ``artifacts``) is rejected rather than unpickled.
    """
    data = torch.load(path, map_location=map_location, weights_only=True)
    target = TargetInfo(**data["target"]) if data["target"] is not None else None
    return State(
        input=data["input"],
        target=target,
        artifacts=data["artifacts"],
        metrics=data["metrics"],
        loss=data["loss"],
        iteration=data["iteration"],
        metadata=RunMetadata(seed=data["metadata_seed"]),
    )
