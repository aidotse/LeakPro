#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Deterministic clique search for Carlini's repeatability graph."""

from __future__ import annotations

import torch
from torch import Tensor


def _population_count(value: int) -> int:
    """Count set bits without relying on Python 3.10's int.bit_count()."""
    return bin(value).count("1")


def adjacency_to_bitsets(adjacency: Tensor) -> list[int]:
    """Convert a symmetric boolean adjacency matrix to integer bitsets."""
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square.")
    if adjacency.dtype is not torch.bool:
        raise TypeError("adjacency must have boolean dtype.")
    if not torch.equal(adjacency, adjacency.transpose(0, 1)):
        raise ValueError("adjacency must be symmetric.")
    masks: list[int] = []
    for row_index in range(adjacency.shape[0]):
        mask = 0
        for column_index, connected in enumerate(adjacency[row_index].tolist()):
            if connected and column_index != row_index:
                mask |= 1 << column_index
        masks.append(mask)
    return masks


def maximum_clique(adjacency: Tensor) -> list[int]:
    """Return an exact maximum clique using deterministic bitset branch-and-bound."""
    neighbors = adjacency_to_bitsets(adjacency)
    vertex_count = len(neighbors)
    best: tuple[int, ...] = ()

    def choose_pivot(candidates: int) -> int:
        vertices = [index for index in range(vertex_count) if candidates & (1 << index)]
        return min(vertices, key=lambda index: (-_population_count(candidates & neighbors[index]), index))

    stack = [((1 << vertex_count) - 1, ())]
    while stack:
        candidates, clique = stack.pop()
        if len(clique) + _population_count(candidates) < len(best):
            continue
        if candidates == 0:
            canonical = tuple(sorted(clique))
            if len(canonical) > len(best) or (len(canonical) == len(best) and canonical < best):
                best = canonical
            continue
        pivot = choose_pivot(candidates)
        extensions = candidates & ~neighbors[pivot]
        children: list[tuple[int, tuple[int, ...]]] = []
        while extensions:
            bit = extensions & -extensions
            vertex = bit.bit_length() - 1
            children.append((candidates & neighbors[vertex], clique + (vertex,)))
            candidates &= ~bit
            extensions &= ~bit
            if len(clique) + _population_count(candidates) < len(best):
                break
        stack.extend(reversed(children))

    return list(best)


def clique_medoid(clique: list[int], distances: Tensor) -> tuple[int, float]:
    """Return the deterministic medoid and mean unordered pairwise clique distance."""
    if not clique:
        raise ValueError("clique must not be empty.")
    index = torch.as_tensor(clique, dtype=torch.long)
    submatrix = distances.index_select(0, index).index_select(1, index)
    mean_distances = submatrix.mean(dim=1)
    tied_positions = torch.nonzero(torch.isclose(mean_distances, mean_distances.min()), as_tuple=False).flatten()
    medoid = min(clique[int(position)] for position in tied_positions.tolist())
    pair_count = len(clique) * (len(clique) - 1) // 2
    mean_pairwise = 0.0 if pair_count == 0 else float(submatrix.triu(diagonal=1).sum() / pair_count)
    return medoid, mean_pairwise
