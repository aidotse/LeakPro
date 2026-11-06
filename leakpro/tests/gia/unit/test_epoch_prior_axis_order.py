#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""EpochOrderInvariantPrior finds the epoch dimension of a seed slice whatever order the axes were declared in."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.losses.regularization import EpochOrderInvariantPrior
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs

SEEDS, EPOCHS, N = 2, 3, 4


def _value(axes: AxisManager, recon_slice: torch.Tensor, slice_axis: str = "seed") -> torch.Tensor:
    inputs = LossInputs(recon=recon_slice, target=None, target_obs=None, ctx=SimpleNamespace(axes=axes),
                        seed_index=0, slice_axis=slice_axis)
    return EpochOrderInvariantPrior().compute(inputs)


def _reference(epoch_major: torch.Tensor) -> torch.Tensor:
    """The prior on one seed's ``[E, N, ...]`` slice: mean squared distance between per-epoch means."""
    means = epoch_major.mean(dim=1)
    pairs = [(means[i] - means[j]).pow(2).sum() for i in range(EPOCHS) for j in range(i + 1, EPOCHS)]
    return torch.stack(pairs).mean()


def test_seed_then_epoch_declaration_uses_the_epoch_dimension() -> None:
    gen = torch.Generator().manual_seed(0)
    full = torch.randn(SEEDS, EPOCHS, N, 1, 2, 2, generator=gen)  # axes declared (seed, epoch)
    recon_slice = full.select(0, 0)  # [E, N, ...]: epoch is now dim 0
    got = _value(AxisManager(Axis("seed", SEEDS), Axis("epoch", EPOCHS)), recon_slice)
    torch.testing.assert_close(got, _reference(recon_slice))


def test_epoch_then_seed_declaration_is_unchanged() -> None:
    gen = torch.Generator().manual_seed(0)
    full = torch.randn(EPOCHS, SEEDS, N, 1, 2, 2, generator=gen)  # axes declared (epoch, seed)
    recon_slice = full.select(1, 0)
    got = _value(AxisManager(Axis("epoch", EPOCHS), Axis("seed", SEEDS)), recon_slice)
    torch.testing.assert_close(got, _reference(recon_slice))


def test_slicing_along_the_epoch_axis_is_rejected() -> None:
    recon_slice = torch.randn(N, 1, 2, 2)
    with pytest.raises(ValueError, match="epoch"):
        _value(AxisManager(Axis("epoch", EPOCHS)), recon_slice, slice_axis="epoch")
