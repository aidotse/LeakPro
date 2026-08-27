"""Deterministic numerical checks for extraction utilities."""

from __future__ import annotations

from itertools import combinations

import pytest
import torch

from leakpro.attacks.extraction_attacks.graph import clique_medoid, maximum_clique
from leakpro.attacks.extraction_attacks.configs import SimilarityBand
from leakpro.attacks.extraction_attacks.metrics import (
    carlini_reference_scores,
    l2_band_scores,
    nearest_reference,
    normalized_l2_pairwise,
    pairwise_band_scores,
    tiled_l2_pairwise,
)


def _brute_maximum_clique(adjacency: torch.Tensor) -> list[int]:
    vertices = list(range(adjacency.shape[0]))
    best: tuple[int, ...] = ()
    for size in range(1, len(vertices) + 1):
        for subset in combinations(vertices, size):
            if all(bool(adjacency[left, right]) for left, right in combinations(subset, 2)):
                if len(subset) > len(best) or (len(subset) == len(best) and subset < best):
                    best = subset
    return list(best)


def test_normalized_l2_pairwise_matches_direct_formula() -> None:
    left = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    right = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 1.0], [1.0, 1.0]]]])
    distances = normalized_l2_pairwise(left, right, block_size=1)
    expected = torch.tensor([[2.0**-0.5, 2.0**-0.5]])
    torch.testing.assert_close(distances, expected)


def test_tiled_l2_uses_the_worst_tile() -> None:
    images = torch.zeros((2, 1, 4, 4))
    images[1, :, :2, :2] = 1.0
    distances = tiled_l2_pairwise(images, tile_grid=(2, 2), block_size=1)
    assert distances[0, 1].item() == 1.0
    assert distances[1, 0].item() == 1.0


def test_near_copy_distances_avoid_high_resolution_cancellation() -> None:
    original = torch.full((1, 3, 512, 512), 0.75)
    near_copy = original + 1e-5
    images = torch.cat((original, near_copy))
    expected = (original - near_copy).square().mean().sqrt()

    normalized = normalized_l2_pairwise(original, near_copy, block_size=1)[0, 0]
    tiled = tiled_l2_pairwise(images, tile_grid=(4, 4), block_size=1)[0, 1]

    torch.testing.assert_close(normalized, expected, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(tiled, expected, rtol=1e-5, atol=1e-8)
    assert normalized < 5e-4
    assert tiled < 5e-4


def test_maximum_clique_matches_brute_force_on_random_graphs() -> None:
    generator = torch.Generator().manual_seed(123)
    for _ in range(20):
        upper = torch.rand((8, 8), generator=generator).lt(0.35).triu(diagonal=1)
        adjacency = upper | upper.transpose(0, 1)
        assert maximum_clique(adjacency) == _brute_maximum_clique(adjacency)


def test_maximum_clique_handles_a_dense_graph_beyond_the_recursion_limit() -> None:
    vertex_count = 1_100
    adjacency = torch.ones((vertex_count, vertex_count), dtype=torch.bool)
    adjacency.fill_diagonal_(False)

    assert maximum_clique(adjacency) == list(range(vertex_count))


def test_clique_medoid_uses_lowest_index_for_a_tie() -> None:
    distances = torch.tensor(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.1],
            [0.2, 0.1, 0.0],
        ]
    )
    medoid, mean_distance = clique_medoid([0, 1, 2], distances)
    assert medoid == 1
    assert mean_distance == torch.tensor(0.4 / 3).item()


def test_carlini_reference_ratio_flags_exact_copy() -> None:
    references = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[0.2, 0.2], [0.2, 0.2]]],
            [[[0.4, 0.4], [0.4, 0.4]]],
        ]
    )
    candidates = references[:1].clone()
    scores = carlini_reference_scores(candidates, references, neighbors=3, alpha=0.5)
    assert scores.nearest_indices.tolist() == [0]
    assert scores.nearest_distances.tolist() == [0.0]
    assert scores.ratios.tolist() == [0.0]


def test_carlini_reference_ratio_uses_nearest_reference_neighborhood() -> None:
    references = torch.tensor([[[[0.0]]], [[[0.2]]], [[[0.4]]], [[[1.0]]]])
    candidates = torch.tensor([[[[0.1]]]])

    scores = carlini_reference_scores(candidates, references, neighbors=3, alpha=0.5, block_size=2)

    assert scores.nearest_indices.tolist() == [0]
    assert scores.neighbor_mean_distances.tolist() == pytest.approx([0.2])
    assert scores.ratios.tolist() == pytest.approx([1.0])
    assert scores.ratios.le(1.0).tolist() == [True]


def test_streamed_reference_metrics_match_full_pairwise_values() -> None:
    generator = torch.Generator().manual_seed(19)
    candidates = torch.rand((5, 1, 3, 3), generator=generator)
    references = torch.rand((7, 1, 3, 3), generator=generator)
    full = normalized_l2_pairwise(candidates, references, block_size=2)
    indices, distances = nearest_reference(candidates, references, block_size=2)
    expected_distances, expected_indices = full.min(dim=1)
    torch.testing.assert_close(distances, expected_distances)
    assert indices.tolist() == expected_indices.tolist()
    band = {"near": SimilarityBand(lower=0.0, upper=float(full.median()))}
    streamed = l2_band_scores(candidates, references, band, block_size=2)
    matches = full.le(band["near"].upper)
    assert streamed["near"]["matched_generations"] == int(matches.any(dim=1).sum())
    assert streamed["near"]["unique_references"] == int(matches.any(dim=0).sum())


def test_generic_pairwise_bands_support_similarity_scores() -> None:
    candidates = torch.tensor([[[[0.0]]], [[[1.0]]]])
    references = torch.tensor([[[[0.0]]], [[[0.5]]], [[[1.0]]]])

    def similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 1.0 - (left.flatten(start_dim=1) - right.flatten(start_dim=1).transpose(0, 1)).abs()

    result = pairwise_band_scores(
        candidates,
        references,
        {"exact": SimilarityBand(lower=0.99, upper=1.0)},
        similarity,
        block_size=1,
    )
    assert result["exact"]["ams"] == 1.0
    assert result["exact"]["ums"] == 1.0


def test_similarity_ams_uses_best_match_while_ums_keeps_in_band_references() -> None:
    candidates = torch.zeros((1, 1, 1, 1))
    references = torch.tensor([[[[0.45]]], [[[0.70]]]])

    def similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return right.flatten(start_dim=1).transpose(0, 1).expand(left.shape[0], -1)

    result = pairwise_band_scores(
        candidates,
        references,
        {"lower": SimilarityBand(lower=0.4, upper=0.5)},
        similarity,
        block_size=1,
    )

    assert result["lower"]["ams"] == 0.0
    assert result["lower"]["ums"] == 1.0


def test_l2_ams_uses_nearest_match_while_ums_keeps_in_band_references() -> None:
    candidates = torch.zeros((1, 1, 1, 1))
    references = torch.tensor([[[[0.10]]], [[[0.45]]]])

    result = l2_band_scores(
        candidates,
        references,
        {"farther": SimilarityBand(lower=0.4, upper=0.5)},
        block_size=1,
    )

    assert result["farther"]["ams"] == 0.0
    assert result["farther"]["ums"] == 1.0
