#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Utilities for matching and assignment problems in gradient inversion attacks.

This module provides reusable functions for solving assignment problems using
the Hungarian algorithm, commonly used for:
- Matching reconstructed images to ground truth (handling permutation invariance)
- Matching reconstructions across epochs in FedAvg attacks
- Computing optimal pairings based on various distance metrics
"""

from __future__ import annotations

from typing import Literal

import torch
from scipy.optimize import linear_sum_assignment


def compute_cost_matrix(
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    metric: Literal["l2", "mse", "psnr"] = "l2",
) -> torch.Tensor:
    """Compute pairwise cost matrix between two sets of images.

    Args:
        images_a: First set of images with shape [N, C, H, W] or [N, ...]
        images_b: Second set of images with shape [N, C, H, W] or [N, ...]
        metric: Distance metric to use
            - "l2": L2 distance (Euclidean)
            - "mse": Mean squared error
            - "psnr": Peak signal-to-noise ratio (negated, so lower is better)

    Returns:
        Cost matrix with shape [N, N] where cost[i, j] is the distance
        between images_a[i] and images_b[j]

    Examples:
        >>> images_a = torch.randn(4, 3, 32, 32)
        >>> images_b = torch.randn(4, 3, 32, 32)
        >>> cost = compute_cost_matrix(images_a, images_b, metric="mse")
        >>> cost.shape
        torch.Size([4, 4])

    """
    if images_a.shape[0] != images_b.shape[0]:
        raise ValueError(
            f"Batch sizes must match: {images_a.shape[0]} vs {images_b.shape[0]}"
        )

    n = images_a.shape[0]
    cost = torch.zeros(n, n, device=images_a.device)

    for i in range(n):
        for j in range(n):
            img_a = images_a[i]
            img_b = images_b[j]

            if metric == "l2":
                cost[i, j] = torch.sqrt(((img_a - img_b) ** 2).sum())
            elif metric == "mse":
                cost[i, j] = ((img_a - img_b) ** 2).mean()
            elif metric == "psnr":
                # For PSNR, lower is better, so we negate it
                mse = ((img_a - img_b) ** 2).mean()
                if mse == 0:
                    cost[i, j] = 0.0
                else:
                    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
                    cost[i, j] = -psnr  # Negate so lower is better
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return cost


def hungarian_matching(cost_matrix: torch.Tensor) -> list[int]:
    """Solve assignment problem using Hungarian algorithm.

    Uses scipy's linear_sum_assignment to find optimal matching that
    minimizes total cost.

    Args:
        cost_matrix: Square cost matrix with shape [N, N]

    Returns:
        List of N assignments where result[i] is the matched index in
        the second set for element i in the first set

    Examples:
        >>> cost = torch.tensor([[1.0, 2.0], [2.0, 0.5]])
        >>> matches = hungarian_matching(cost)
        >>> matches
        [0, 1]  # Element 0 matches to 0, element 1 matches to 1

    References:
        Kuhn, H. W. (1955). The Hungarian method for the assignment problem.
        Naval Research Logistics Quarterly, 2(1-2), 83-97.

    """
    if cost_matrix.ndim != 2:
        raise ValueError(
            f"Expected 2D cost matrix, got {cost_matrix.ndim}D: {cost_matrix.shape}"
        )

    if cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError(
            f"Cost matrix must be square, got {cost_matrix.shape}"
        )

    # Convert to numpy for scipy
    cost_np = cost_matrix.detach().cpu().numpy()

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_np)

    # row_ind should be [0, 1, 2, ...], col_ind contains the matching
    return col_ind.tolist()


def match_images(
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    metric: Literal["l2", "mse", "psnr"] = "mse",
) -> tuple[list[int], torch.Tensor]:
    """Match images optimally using Hungarian algorithm.

    Convenience function that combines cost matrix computation and matching.
    Useful for matching reconstructed images to ground truth.

    Args:
        images_a: First set of images with shape [N, C, H, W]
        images_b: Second set of images with shape [N, C, H, W]
        metric: Distance metric ("l2", "mse", or "psnr")

    Returns:
        Tuple of (matches, cost_matrix) where:
            - matches: List of N assignments
            - cost_matrix: The computed cost matrix [N, N]

    Examples:
        >>> reconstructions = torch.randn(4, 3, 32, 32)
        >>> ground_truth = torch.randn(4, 3, 32, 32)
        >>> matches, costs = match_images(reconstructions, ground_truth)
        >>> matched_reconstructions = reconstructions[matches]

    """
    cost_matrix = compute_cost_matrix(images_a, images_b, metric=metric)
    matches = hungarian_matching(cost_matrix)
    return matches, cost_matrix


__all__ = [
    "compute_cost_matrix",
    "hungarian_matching",
    "match_images",
]
