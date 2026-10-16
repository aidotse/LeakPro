"""Deterministic numerical checks for extraction utilities."""

from __future__ import annotations

from itertools import combinations
from time import perf_counter

import pytest
import torch

from leakpro.attacks.extraction_attacks.utils_generative.carlini_graph import maximum_clique
from leakpro.attacks.extraction_attacks.utils_generative.image_metrics import (
    SimilarityBand,
    carlini_reference_scores,
    l2_band_scores,
    nearest_reference,
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


def test_nearest_reference_matches_direct_formula() -> None:
    left = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    right = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 1.0], [1.0, 1.0]]]])
    indices, distances = nearest_reference(left, right, block_size=1)
    expected = torch.tensor([2.0**-0.5])
    assert indices.tolist() == [0]
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

    normalized = nearest_reference(original, near_copy, block_size=1)[1][0]
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


def test_maximum_clique_preserves_the_selected_group_when_sizes_tie() -> None:
    adjacency = torch.zeros((8, 8), dtype=torch.bool)
    edges = [(0, 3), (0, 5), (0, 6), (0, 7), (1, 2), (1, 5), (1, 6), (2, 3), (2, 4),
             (2, 5), (2, 7), (3, 4), (3, 6), (4, 5), (4, 6), (4, 7), (5, 6), (6, 7)]
    for left, right in edges:
        adjacency[left, right] = adjacency[right, left] = True

    assert maximum_clique(adjacency) == [0, 3, 6]


def test_maximum_clique_prunes_many_equal_dense_choices() -> None:
    parts = 24
    adjacency = torch.ones((2 * parts, 2 * parts), dtype=torch.bool)
    adjacency.fill_diagonal_(False)
    for pair in range(parts):
        adjacency[2 * pair, 2 * pair + 1] = False
        adjacency[2 * pair + 1, 2 * pair] = False

    start = perf_counter()
    assert maximum_clique(adjacency) == list(range(0, 2 * parts, 2))
    assert perf_counter() - start < 10.0


def test_maximum_clique_budget_warns_and_rejects_an_unfinished_search(caplog: pytest.LogCaptureFixture) -> None:
    adjacency = torch.zeros((3, 3), dtype=torch.bool)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="budget of 2 search nodes.*No exact result was returned"):
            maximum_clique(adjacency, max_search_nodes=2)

    assert len(caplog.records) == 2
    assert all(record.levelname == "WARNING" for record in caplog.records)
    assert caplog.records[0].message == caplog.records[1].message
    assert "increase max_clique_search_nodes" in caplog.text

    # Four popped states finish this graph, retaining the original tie break.
    caplog.clear()
    assert maximum_clique(adjacency, max_search_nodes=4) == _brute_maximum_clique(adjacency) == [0]
    assert not caplog.records


@pytest.mark.parametrize(
    ("budget", "error"), [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError), (None, TypeError)],
)
def test_maximum_clique_rejects_invalid_budgets(budget: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="positive integer"):
        maximum_clique(torch.zeros((1, 1), dtype=torch.bool), max_search_nodes=budget)


def test_carlini_reference_ratio_uses_generated_image_neighborhood() -> None:
    references = torch.tensor([[[[0.0]]], [[[0.2]]], [[[0.4]]], [[[1.0]]]])
    candidates = torch.tensor([[[[0.1]]]])

    scores = carlini_reference_scores(candidates, references, neighbors=3, alpha=0.5, block_size=2)

    assert scores.nearest_indices.tolist() == [0]
    assert scores.neighbor_mean_distances.tolist() == pytest.approx([1 / 6])
    assert scores.ratios.tolist() == pytest.approx([1.2])
    assert scores.ratios.le(1.0).tolist() == [False]


def test_streamed_reference_metrics_match_full_pairwise_values(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = torch.Generator().manual_seed(19)
    candidates = torch.rand((5, 1, 3, 3), generator=generator)
    references = torch.rand((7, 1, 3, 3), generator=generator)
    full = (candidates[:, None] - references[None, :]).square().flatten(2).mean(2).sqrt()
    indices, distances = nearest_reference(candidates, references, block_size=2)
    expected_distances, expected_indices = full.min(dim=1)
    torch.testing.assert_close(distances, expected_distances)
    assert indices.tolist() == expected_indices.tolist()
    band = {"near": SimilarityBand(lower=0.0, upper=float(full.median()))}
    from leakpro.attacks.extraction_attacks.utils_generative import image_metrics

    original = image_metrics._normalized_l2_block
    evaluated_pairs = 0

    def counted(left: torch.Tensor, right: torch.Tensor, dimensions: int) -> torch.Tensor:
        nonlocal evaluated_pairs
        evaluated_pairs += left.shape[0] * right.shape[0]
        return original(left, right, dimensions)

    monkeypatch.setattr(image_metrics, "_normalized_l2_block", counted)
    combined_indices, combined_distances, streamed = l2_band_scores(
        candidates, references, band, block_size=2, return_nearest=True,
    )
    assert evaluated_pairs == candidates.shape[0] * references.shape[0]
    assert combined_indices.tolist() == indices.tolist()
    torch.testing.assert_close(combined_distances, distances)
    matches = full.le(band["near"].upper)
    assert streamed["near"]["matched_generations"] == int(matches.any(dim=1).sum())
    assert streamed["near"]["unique_references"] == int(matches.any(dim=0).sum())


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

    def similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 1.0 - (left[:, None] - right[None, :]).square().flatten(2).mean(2).sqrt()

    scores = pairwise_band_scores(candidates, references,
                                 {"lower": SimilarityBand(lower=0.5, upper=0.6)}, similarity)
    assert scores["lower"]["ams"] == 0.0
    assert scores["lower"]["ums"] == 1.0
