#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Paper-aligned image distances and extraction metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor

from leakpro.attacks.extraction_attacks.protocols import PairwiseScore
from leakpro.attacks.extraction_attacks.utils_generative import batch_ranges
from leakpro.utils.device import get_device
from leakpro.utils.import_helper import Self


class SimilarityBand(BaseModel):
    """Inclusive band used for SIDE's AMS and UMS metrics."""

    lower: float
    upper: float
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Reject inverted bands."""
        if self.lower > self.upper:
            raise ValueError("SimilarityBand.lower must not exceed upper.")
        return self


@dataclass(frozen=True)
class ReferenceScores:
    """Nearest-reference distances and Carlini adaptive ratios."""

    nearest_indices: Tensor
    nearest_distances: Tensor
    neighbor_mean_distances: Tensor
    ratios: Tensor


def _flatten_float(images: Tensor, device: torch.device) -> Tensor:
    return images.to(device=device, dtype=torch.float32).flatten(start_dim=1)


def _validate_pairwise_inputs(left: Tensor, right: Tensor, block_size: int, neighbors: int | None = None) -> None:
    if left.ndim != 4 or right.ndim != 4 or tuple(left.shape[1:]) != tuple(right.shape[1:]):
        raise ValueError("left and right must be BCHW tensors with identical image shapes.")
    if left.shape[0] < 1 or right.shape[0] < 1:
        raise ValueError("left and right must both contain at least one image.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    if neighbors is not None:
        if neighbors < 1:
            raise ValueError("neighbors must be positive.")
        if right.shape[0] < neighbors:
            raise ValueError(f"reference set has {right.shape[0]} images but neighbors={neighbors}.")


def _normalized_l2_block(left: Tensor, right: Tensor, dimensions: int) -> Tensor:
    """Compute direct squared differences in feature blocks to avoid cancellation."""
    squared = torch.zeros((left.shape[0], right.shape[0]), dtype=torch.float32, device=left.device)
    feature_block_size = min(2_048, dimensions)
    for start, end in batch_ranges(dimensions, feature_block_size):
        difference = left[:, None, start:end] - right[None, :, start:end]
        squared += difference.square().sum(dim=2)
    return squared.div(float(dimensions)).sqrt()


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
    target_device = get_device() if device == "auto" else torch.device(device)
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


@dataclass(frozen=True)
class _ReferenceStatistics:
    indices: Tensor
    scores: Tensor
    neighbor_means: Tensor | None
    bands: dict[str, dict[str, float | int]]


def _reference_statistics(
    candidates: Tensor,
    references: Tensor,
    *,
    block_size: int,
    device: str,
    bands: dict[str, SimilarityBand] | None = None,
    neighbors: int | None = None,
    score_fn: PairwiseScore | None = None,
    higher_is_better: bool = False,
) -> _ReferenceStatistics:
    """Accumulate nearest scores, top-k means, and band hits in one blocked pass."""
    _validate_pairwise_inputs(candidates, references, block_size, neighbors)
    bands = {} if bands is None else bands
    target_device = get_device() if device == "auto" else torch.device(device)
    dimensions = candidates[0].numel()
    reference_matches = {name: torch.zeros(references.shape[0], dtype=torch.bool, device="cpu") for name in bands}
    indices, scores, means = [], [], []
    initial = -torch.inf if higher_is_better else torch.inf
    for start, end in batch_ranges(candidates.shape[0], block_size):
        left = (_flatten_float(candidates[start:end], target_device) if score_fn is None
                else candidates[start:end].to(target_device))
        best = torch.full((end - start,), initial, dtype=torch.float32, device=target_device)
        best_indices = torch.full((end - start,), -1, dtype=torch.long, device=target_device)
        nearest = torch.empty((end - start, 0), device=target_device)
        for ref_start, ref_end in batch_ranges(references.shape[0], block_size):
            right = (_flatten_float(references[ref_start:ref_end], target_device) if score_fn is None
                     else references[ref_start:ref_end].to(target_device))
            if score_fn is None:
                block = _normalized_l2_block(left, right, dimensions)
            else:
                block = score_fn(left, right)
                expected_shape = (end - start, ref_end - ref_start)
                if not isinstance(block, Tensor) or tuple(block.shape) != expected_shape:
                    raise ValueError(f"score_fn must return shape {expected_shape}.")
                if not torch.isfinite(block).all():
                    raise ValueError("score_fn returned NaN or infinity.")
                block = block.to(device=target_device, dtype=torch.float32)
            block_best, local_indices = block.max(dim=1) if higher_is_better else block.min(dim=1)
            improve = block_best > best if higher_is_better else block_best < best
            best_indices = torch.where(improve, local_indices + ref_start, best_indices)
            best = torch.maximum(best, block_best) if higher_is_better else torch.minimum(best, block_best)
            if neighbors is not None:
                combined = torch.cat((nearest, block), dim=1)
                nearest = combined.topk(min(neighbors, combined.shape[1]), dim=1,
                                        largest=higher_is_better, sorted=False).values
            for name, band in bands.items():
                matches = block.ge(band.lower) & block.le(band.upper)
                reference_matches[name][ref_start:ref_end] |= matches.any(dim=0).cpu()
        indices.append(best_indices.cpu())
        scores.append(best.cpu())
        if neighbors is not None:
            means.append(nearest.mean(dim=1).cpu())
    best_scores = torch.cat(scores)
    metrics = {}
    for name, band in bands.items():
        matched = int((best_scores.ge(band.lower) & best_scores.le(band.upper)).sum())
        unique = int(reference_matches[name].sum())
        metrics[name] = {"matched_generations": matched, "unique_references": unique,
                         "ams": matched / candidates.shape[0], "ums": unique / candidates.shape[0]}
    return _ReferenceStatistics(torch.cat(indices), best_scores, torch.cat(means) if means else None, metrics)


def carlini_reference_scores(
    candidates: Tensor,
    references: Tensor,
    *,
    neighbors: int = 50,
    alpha: float = 0.5,
    block_size: int = 64,
    device: str = "cpu",
) -> ReferenceScores:
    """Compare each candidate's nearest distance with its own k-neighbor mean."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    stats = _reference_statistics(candidates, references, block_size=block_size, device=device, neighbors=neighbors)
    if stats.neighbor_means is None:
        raise RuntimeError("Reference-neighbor means were not computed.")
    denominator = alpha * stats.neighbor_means
    if torch.any(denominator <= 0):
        raise ValueError("Candidate-neighbor mean distance is zero; the adaptive ratio is undefined.")
    return ReferenceScores(stats.indices, stats.scores, stats.neighbor_means, stats.scores / denominator)


def nearest_reference(
    candidates: Tensor,
    references: Tensor,
    *,
    block_size: int = 64,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Return each candidate's nearest reference index and normalized L2 distance."""
    stats = _reference_statistics(candidates, references, block_size=block_size, device=device)
    return stats.indices, stats.scores


def l2_band_scores(
    candidates: Tensor,
    references: Tensor,
    bands: dict[str, SimilarityBand],
    *,
    block_size: int = 64,
    device: str = "cpu",
    return_nearest: bool = False,
) -> dict[str, dict[str, float | int]] | tuple[Tensor, Tensor, dict[str, dict[str, float | int]]]:
    """Compute L2 AMS/UMS and optionally return nearest matches from the same pass.

    AMS classifies each generation by its nearest-reference distance. UMS
    counts all references reached by any in-band pair.
    """
    if not bands and not return_nearest:
        return {}
    stats = _reference_statistics(candidates, references, block_size=block_size, device=device, bands=bands)
    if return_nearest:
        return stats.indices, stats.scores, stats.bands
    return stats.bands


def pairwise_band_scores(
    candidates: Tensor,
    references: Tensor,
    bands: dict[str, SimilarityBand],
    score_fn: PairwiseScore,
    *,
    block_size: int = 64,
    device: str = "cpu",
) -> dict[str, dict[str, float | int]]:
    """Compute AMS/UMS for a finite higher-is-more-similar pairwise score.

    The callback receives BCHW blocks and returns a candidate-by-reference
    matrix. AMS uses each candidate's maximum; UMS retains all in-band hits.
    """
    if not bands:
        return {}
    return _reference_statistics(candidates, references, block_size=block_size, device=device,
                                 bands=bands, score_fn=score_fn, higher_is_better=True).bands
