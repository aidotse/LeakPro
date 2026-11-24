#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Shared Hungarian (optimal bipartite) row-matching utilities.

Used by ``components/evaluators/image.py`` (aligning a reconstructed image batch
to its ground truth), ``components/aggregators.py::EpochMatchingAggregator``
(aligning epoch reconstructions to epoch 0), and any evaluator that aligns an
order-less reconstruction to its ground truth.

``hungarian_match`` accepts rectangular cost matrices, so an evaluator can report a
row-count mismatch instead of discarding rows to force a square problem.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def hungarian_match(cost: Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the linear assignment problem for a ``[N, N]`` cost matrix.

    Accepts either a Tensor or an ndarray. Returns ``(row_ind, col_ind)`` exactly
    as ``scipy.optimize.linear_sum_assignment`` does — for a square cost matrix
    ``row_ind`` is always sorted ``0..N-1``, so ``col_ind[i]`` is the matched
    column for row ``i``.
    """
    if isinstance(cost, Tensor):
        cost = cost.detach().cpu().numpy()
    return linear_sum_assignment(cost)


def hungarian_align(target: Tensor, recon: Tensor) -> tuple[Tensor, np.ndarray]:
    """Align ``recon`` rows to ``target`` rows via minimum squared-L2 bipartite matching.

    ``target`` and ``recon`` are ``[N, ...]`` tensors with matching shape; flattened
    to ``[N, D]`` for the cost computation. Returns ``(aligned_recon, col_ind)``
    where ``aligned_recon[i]`` corresponds to ``target[i]``.
    """
    t_flat = target.reshape(target.shape[0], -1).float()
    r_flat = recon.reshape(recon.shape[0], -1).float()
    cost = torch.cdist(t_flat.unsqueeze(0), r_flat.unsqueeze(0), p=2).squeeze(0) ** 2
    _, col_ind = hungarian_match(cost)
    return recon[col_ind], col_ind
