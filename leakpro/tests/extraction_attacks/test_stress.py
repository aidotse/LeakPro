"""Moderate scaling checks for blocked distances and Carlini graph construction."""

from __future__ import annotations

import torch

from leakpro.attacks.extraction_attacks.metrics import (
    carlini_reference_scores,
    normalized_l2_pairwise,
    tiled_l2_pairwise,
)


def test_reference_scoring_handles_nondivisible_stress_blocks() -> None:
    generator = torch.Generator().manual_seed(20260826)
    candidates = torch.rand((37, 3, 8, 8), generator=generator)
    references = torch.rand((53, 3, 8, 8), generator=generator)

    scores = carlini_reference_scores(
        candidates,
        references,
        neighbors=17,
        alpha=0.5,
        block_size=11,
    )
    dense = normalized_l2_pairwise(candidates, references, block_size=13)
    expected_distances, expected_indices = dense.min(dim=1)

    assert torch.equal(scores.nearest_indices, expected_indices)
    assert torch.allclose(scores.nearest_distances, expected_distances, atol=1e-6)
    assert scores.ratios.shape == (37,)
    assert torch.isfinite(scores.ratios).all()


def test_tiled_distance_stress_matrix_is_symmetric_and_finite() -> None:
    generator = torch.Generator().manual_seed(17)
    images = torch.rand((61, 3, 16, 16), generator=generator)

    distances = tiled_l2_pairwise(images, tile_grid=(4, 4), block_size=13)

    assert distances.shape == (61, 61)
    assert torch.allclose(distances, distances.transpose(0, 1), atol=1e-6)
    assert torch.equal(distances.diagonal(), torch.zeros(61))
    assert torch.isfinite(distances).all()
