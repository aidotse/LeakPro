#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Paper-aligned image distances and extraction metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from leakpro.attacks.extraction_attacks.configs import SimilarityBand
from leakpro.attacks.extraction_attacks.protocols import PairwiseScore
from leakpro.attacks.extraction_attacks.utils import batch_ranges, resolve_device


@dataclass(frozen=True)
class ReferenceScores:
    """Nearest-reference distances and Carlini adaptive ratios."""

    nearest_indices: Tensor
    nearest_distances: Tensor
    neighbor_mean_distances: Tensor
    ratios: Tensor


def _flatten_float(images: Tensor, device: torch.device) -> Tensor:
    return images.to(device=device, dtype=torch.float32).flatten(start_dim=1)


def _validate_pairwise_inputs(left: Tensor, right: Tensor, block_size: int) -> None:
    if left.ndim != 4 or right.ndim != 4 or tuple(left.shape[1:]) != tuple(right.shape[1:]):
        raise ValueError("left and right must be BCHW tensors with identical image shapes.")
    if left.shape[0] < 1 or right.shape[0] < 1:
        raise ValueError("left and right must both contain at least one image.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")


def _normalized_l2_block(left: Tensor, right: Tensor, dimensions: int) -> Tensor:
    """Compute direct squared differences in feature blocks to avoid cancellation."""
    squared = torch.zeros((left.shape[0], right.shape[0]), dtype=torch.float32, device=left.device)
    feature_block_size = min(2_048, dimensions)
    for start, end in batch_ranges(dimensions, feature_block_size):
        difference = left[:, None, start:end] - right[None, :, start:end]
        squared += difference.square().sum(dim=2)
    return squared.div(float(dimensions)).sqrt()


def normalized_l2_pairwise(
    left: Tensor,
    right: Tensor,
    *,
    block_size: int = 64,
    device: str = "cpu",
) -> Tensor:
    """Compute sqrt(mean squared pixel error) for every image pair."""
    _validate_pairwise_inputs(left, right, block_size)
    target_device = resolve_device(device)
    dimensions = left[0].numel()
    result = torch.empty((left.shape[0], right.shape[0]), dtype=torch.float32)
    for left_start, left_end in batch_ranges(left.shape[0], block_size):
        left_block = _flatten_float(left[left_start:left_end], target_device)
        for right_start, right_end in batch_ranges(right.shape[0], block_size):
            right_block = _flatten_float(right[right_start:right_end], target_device)
            distances = _normalized_l2_block(left_block, right_block, dimensions)
            result[left_start:left_end, right_start:right_end] = distances.cpu()
    return result


def tiled_l2_pairwise(
    images: Tensor,
    *,
    tile_grid: tuple[int, int] = (4, 4),
    block_size: int = 64,
    device: str = "cpu",
) -> Tensor:
    """Compute Carlini's maximum normalized L2 distance over non-overlapping tiles."""
    if images.ndim != 4:
        raise ValueError("images must be a BCHW tensor.")
    _, channels, height, width = images.shape
    grid_height, grid_width = tile_grid
    if height % grid_height != 0 or width % grid_width != 0:
        raise ValueError(f"Image shape {(height, width)} is not divisible by tile_grid {tile_grid}.")
    tile_height = height // grid_height
    tile_width = width // grid_width
    tiles = (
        images.reshape(images.shape[0], channels, grid_height, tile_height, grid_width, tile_width)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(images.shape[0], grid_height * grid_width, channels * tile_height * tile_width)
    )
    tile_dimensions = tiles.shape[-1]
    target_device = resolve_device(device)
    result = torch.empty((images.shape[0], images.shape[0]), dtype=torch.float32)
    for left_start, left_end in batch_ranges(images.shape[0], block_size):
        left = tiles[left_start:left_end].to(device=target_device, dtype=torch.float32)
        block_max = torch.zeros((left.shape[0], images.shape[0]), device=target_device)
        for right_start, right_end in batch_ranges(images.shape[0], block_size):
            right = tiles[right_start:right_end].to(device=target_device, dtype=torch.float32)
            tile_max = torch.zeros((left.shape[0], right.shape[0]), device=target_device)
            for tile_index in range(tiles.shape[1]):
                left_tile = left[:, tile_index]
                right_tile = right[:, tile_index]
                distance = _normalized_l2_block(left_tile, right_tile, tile_dimensions)
                tile_max = torch.maximum(tile_max, distance)
            block_max[:, right_start:right_end] = tile_max
        result[left_start:left_end] = block_max.cpu()
    result.fill_diagonal_(0.0)
    return result


def carlini_reference_scores(
    candidates: Tensor,
    references: Tensor,
    *,
    neighbors: int = 50,
    alpha: float = 0.5,
    block_size: int = 64,
    device: str = "cpu",
    reference_neighbor_means: Tensor | None = None,
) -> ReferenceScores:
    """Compute Carlini's reference-centric adaptive ratio from Section 5.1."""
    _validate_pairwise_inputs(candidates, references, block_size)
    if neighbors < 1:
        raise ValueError("neighbors must be positive.")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    if references.shape[0] < neighbors:
        raise ValueError(f"reference set has {references.shape[0]} images but neighbors={neighbors}.")
    if reference_neighbor_means is None:
        reference_neighbor_means = reference_neighborhood_means(
            references,
            neighbors=neighbors,
            block_size=block_size,
            device=device,
        )
    if tuple(reference_neighbor_means.shape) != (references.shape[0],):
        raise ValueError("reference_neighbor_means must contain one value per reference image.")
    if not torch.isfinite(reference_neighbor_means).all() or torch.any(reference_neighbor_means < 0):
        raise ValueError("reference_neighbor_means must be finite and non-negative.")
    nearest_indices, nearest_distances = nearest_reference(
        candidates,
        references,
        block_size=block_size,
        device=device,
    )
    neighbor_means = reference_neighbor_means.detach().cpu()[nearest_indices]
    denominator = alpha * neighbor_means
    if torch.any(denominator <= 0):
        raise ValueError("Reference-neighbor mean distance is zero; the adaptive ratio is undefined.")
    return ReferenceScores(
        nearest_indices=nearest_indices,
        nearest_distances=nearest_distances,
        neighbor_mean_distances=neighbor_means,
        ratios=nearest_distances / denominator,
    )


def reference_neighborhood_means(
    references: Tensor,
    *,
    neighbors: int = 50,
    block_size: int = 64,
    device: str = "cpu",
) -> Tensor:
    """Mean distance from each reference to its k closest references, including itself."""
    _validate_pairwise_inputs(references, references, block_size)
    if neighbors < 1:
        raise ValueError("neighbors must be positive.")
    if references.shape[0] < neighbors:
        raise ValueError(f"reference set has {references.shape[0]} images but neighbors={neighbors}.")
    target_device = resolve_device(device)
    dimensions = references[0].numel()
    means = torch.empty(references.shape[0], dtype=torch.float32)
    for left_start, left_end in batch_ranges(references.shape[0], block_size):
        left = _flatten_float(references[left_start:left_end], target_device)
        nearest = torch.empty((left.shape[0], 0), device=target_device)
        for right_start, right_end in batch_ranges(references.shape[0], block_size):
            right = _flatten_float(references[right_start:right_end], target_device)
            distances = _normalized_l2_block(left, right, dimensions)
            combined = torch.cat((nearest, distances), dim=1)
            keep = min(neighbors, combined.shape[1])
            nearest = torch.topk(combined, k=keep, dim=1, largest=False, sorted=False).values
        means[left_start:left_end] = nearest.mean(dim=1).cpu()
    return means


def nearest_reference(
    candidates: Tensor,
    references: Tensor,
    *,
    block_size: int = 64,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Return each candidate's nearest reference index and normalized L2 distance."""
    _validate_pairwise_inputs(candidates, references, block_size)
    target_device = resolve_device(device)
    dimensions = candidates[0].numel()
    all_indices: list[Tensor] = []
    all_distances: list[Tensor] = []
    for candidate_start, candidate_end in batch_ranges(candidates.shape[0], block_size):
        candidate_block = _flatten_float(candidates[candidate_start:candidate_end], target_device)
        best_distances = torch.full((candidate_block.shape[0],), torch.inf, device=target_device)
        best_indices = torch.full((candidate_block.shape[0],), -1, dtype=torch.long, device=target_device)
        for reference_start, reference_end in batch_ranges(references.shape[0], block_size):
            reference_block = _flatten_float(references[reference_start:reference_end], target_device)
            block_distances = _normalized_l2_block(candidate_block, reference_block, dimensions)
            block_best, local_indices = block_distances.min(dim=1)
            improve = block_best < best_distances
            best_distances = torch.where(improve, block_best, best_distances)
            best_indices = torch.where(improve, local_indices + reference_start, best_indices)
        all_indices.append(best_indices.cpu())
        all_distances.append(best_distances.cpu())
    return torch.cat(all_indices), torch.cat(all_distances)


def l2_band_scores(
    candidates: Tensor,
    references: Tensor,
    bands: dict[str, SimilarityBand],
    *,
    block_size: int = 64,
    device: str = "cpu",
) -> dict[str, dict[str, float | int]]:
    """Compute SIDE AMS/UMS with raw normalized-L2 distance bands.

    AMS classifies each generation by its nearest-reference distance. UMS is the
    number of unique reference records reached by any in-band pair divided by
    N_G, matching equations 6-9 of SIDE. Smaller L2 means greater similarity.
    """
    if not bands:
        return {}
    _validate_pairwise_inputs(candidates, references, block_size)
    generation_count = candidates.shape[0]
    target_device = resolve_device(device)
    dimensions = candidates[0].numel()
    candidate_best = torch.full((generation_count,), torch.inf)
    reference_matches = {
        name: torch.zeros(references.shape[0], dtype=torch.bool)
        for name in bands
    }
    for candidate_start, candidate_end in batch_ranges(generation_count, block_size):
        candidate_block = _flatten_float(candidates[candidate_start:candidate_end], target_device)
        block_best = torch.full((candidate_end - candidate_start,), torch.inf, device=target_device)
        for reference_start, reference_end in batch_ranges(references.shape[0], block_size):
            reference_block = _flatten_float(references[reference_start:reference_end], target_device)
            distances = _normalized_l2_block(candidate_block, reference_block, dimensions)
            block_best = torch.minimum(block_best, distances.min(dim=1).values)
            for name, band in bands.items():
                matches = distances.ge(band.lower) & distances.le(band.upper)
                reference_matches[name][reference_start:reference_end] |= matches.any(dim=0).cpu()
        candidate_best[candidate_start:candidate_end] = block_best.cpu()
    results: dict[str, dict[str, float | int]] = {}
    for name, band in bands.items():
        matched_generations = int((candidate_best.ge(band.lower) & candidate_best.le(band.upper)).sum())
        unique_references = int(reference_matches[name].sum())
        results[name] = {
            "matched_generations": matched_generations,
            "unique_references": unique_references,
            "ams": matched_generations / generation_count,
            "ums": unique_references / generation_count,
        }
    return results


def pairwise_band_scores(
    candidates: Tensor,
    references: Tensor,
    bands: dict[str, SimilarityBand],
    score_fn: PairwiseScore,
    *,
    block_size: int = 64,
    device: str = "cpu",
) -> dict[str, dict[str, float | int]]:
    """Compute AMS/UMS for inclusive bands of a supplied pairwise score.

    ``score_fn`` receives candidate and reference BCHW blocks and must return a
    finite higher-is-more-similar matrix shaped ``(candidate_count,
    reference_count)``. AMS classifies each candidate by its maximum score;
    UMS retains every reference reached by any in-band pair.
    """
    if not bands:
        return {}
    _validate_pairwise_inputs(candidates, references, block_size)
    target_device = resolve_device(device)
    candidate_best = torch.full((candidates.shape[0],), -torch.inf)
    reference_matches = {
        name: torch.zeros(references.shape[0], dtype=torch.bool)
        for name in bands
    }
    for candidate_start, candidate_end in batch_ranges(candidates.shape[0], block_size):
        candidate_block = candidates[candidate_start:candidate_end].to(target_device)
        block_best = torch.full((candidate_end - candidate_start,), -torch.inf, device=target_device)
        for reference_start, reference_end in batch_ranges(references.shape[0], block_size):
            reference_block = references[reference_start:reference_end].to(target_device)
            scores = score_fn(candidate_block, reference_block)
            expected_shape = (candidate_end - candidate_start, reference_end - reference_start)
            if not isinstance(scores, Tensor) or tuple(scores.shape) != expected_shape:
                raise ValueError(f"score_fn must return shape {expected_shape}.")
            if not torch.isfinite(scores).all():
                raise ValueError("score_fn returned NaN or infinity.")
            scores = scores.to(device=target_device, dtype=torch.float32)
            block_best = torch.maximum(block_best, scores.max(dim=1).values)
            for name, band in bands.items():
                matches = scores.ge(band.lower) & scores.le(band.upper)
                reference_matches[name][reference_start:reference_end] |= matches.any(dim=0).cpu()
        candidate_best[candidate_start:candidate_end] = block_best.cpu()
    results: dict[str, dict[str, float | int]] = {}
    for name, band in bands.items():
        matched_generations = int((candidate_best.ge(band.lower) & candidate_best.le(band.upper)).sum())
        unique_references = int(reference_matches[name].sum())
        results[name] = {
            "matched_generations": matched_generations,
            "unique_references": unique_references,
            "ams": matched_generations / candidates.shape[0],
            "ums": unique_references / candidates.shape[0],
        }
    return results
