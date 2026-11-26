#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Deterministic clique search for Carlini's repeatability graph."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from leakpro.utils.logger import logger

DEFAULT_MAX_CLIQUE_SEARCH_NODES = 2_000_000
CLIQUE_SEARCH_POLICY = "bounded_first_maximum_v1"


class CliqueSearchResult(NamedTuple):
    """A valid clique, whether its maximum size was proved, and work spent."""

    clique: list[int]
    exact: bool
    search_nodes: int


def _population_count(value: int) -> int:
    """Count set bits on Python versions without int.bit_count()."""
    return bin(value).count("1")


_population_count = getattr(int, "bit_count", _population_count)


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


def _color_bound(candidates: int, neighbors: list[int]) -> int:
    """Bound clique size by greedily coloring the candidate subgraph."""
    colors = 0
    uncolored = candidates
    while uncolored:
        colors += 1
        available = uncolored
        while available:
            bit = available & -available
            vertex = bit.bit_length() - 1
            uncolored &= ~bit
            available &= ~(bit | neighbors[vertex])
    return colors


def _can_improve(candidates: int, clique: tuple[int, ...], best: tuple[int, ...], neighbors: list[int]) -> bool:
    """Keep only branches that can improve size; retain the first maximum on ties."""
    possible_size = len(clique) + _population_count(candidates)
    if best and possible_size > len(best):
        possible_size = len(clique) + _color_bound(candidates, neighbors)
    return possible_size > len(best)


def maximum_clique(  # noqa: C901 - keep the bounded traversal and incumbent updates together
    adjacency: Tensor, *, max_search_nodes: int = DEFAULT_MAX_CLIQUE_SEARCH_NODES,
) -> CliqueSearchResult:
    """Return the first maximum, or a valid lower bound on budget exhaustion.

    Count popped branch-and-bound states, including pruned states, rather than elapsed time.
    Traversal is deterministic. Equal-size cliques do not replace the incumbent.
    """
    if isinstance(max_search_nodes, bool) or not isinstance(max_search_nodes, int):
        raise TypeError("max_search_nodes must be a positive integer.")
    if max_search_nodes < 1:
        raise ValueError("max_search_nodes must be a positive integer.")
    neighbors = adjacency_to_bitsets(adjacency)
    vertex_count = len(neighbors)
    best: tuple[int, ...] = ()

    def choose_pivot(candidates: int) -> int:
        remaining = candidates
        selected, best_degree = -1, -1
        while remaining:
            bit = remaining & -remaining
            vertex = bit.bit_length() - 1
            degree = _population_count(candidates & neighbors[vertex])
            if degree > best_degree:
                selected, best_degree = vertex, degree
            remaining ^= bit
        return selected

    stack = [((1 << vertex_count) - 1, ())]
    search_nodes = 0
    while stack:
        if search_nodes >= max_search_nodes:
            message = (
                f"Maximum-clique search exhausted its budget of {max_search_nodes} search nodes "
                f"for a {vertex_count}-vertex graph. Returning a lower bound of {len(best)}; "
                "increase max_clique_search_nodes to allow more search work."
            )
            logger.warning(message)
            return CliqueSearchResult(list(best), False, search_nodes)
        search_nodes += 1
        candidates, clique = stack.pop()
        if not _can_improve(candidates, clique, best, neighbors):
            continue
        pivot = choose_pivot(candidates)
        extensions = candidates & ~neighbors[pivot]
        children: list[tuple[int, tuple[int, ...]]] = []
        while extensions:
            bit = extensions & -extensions
            vertex = bit.bit_length() - 1
            extended = clique + (vertex,)
            # Every constructed clique is a witness, even if the cap prevents its pop.
            if len(extended) > len(best):
                best = tuple(sorted(extended))
            children.append((candidates & neighbors[vertex], extended))
            candidates &= ~bit
            extensions &= ~bit
            if not _can_improve(candidates, clique, best, neighbors):
                break
        stack.extend(reversed(children))

    return CliqueSearchResult(list(best), True, search_nodes)


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
