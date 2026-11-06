#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Plots shared by the image examples."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from torch import Tensor  # noqa: E402


def denormalize(tensor: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Map a normalised batch back to pixels: ``tensor * std + mean`` (per-channel ``mean`` and ``std``)."""
    return tensor * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def _img(t: Tensor):  # noqa: ANN202
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def visualize_multiple_attacks(
    originals: Tensor,
    reconstructions: dict[str, Tensor],
    labels: Tensor,
    path: Path,
    *,
    captions: dict[str, str] | None = None,
) -> Path:
    """Grid of the originals (top row) against one row per attack; all tensors in data space ``[N, C, H, W]``.

    ``captions`` adds a line of text (for example the mean PSNR) in front of each attack's row.
    """
    n, rows = originals.shape[0], list(reconstructions)
    fig, axes = plt.subplots(len(rows) + 1, n + 1, figsize=(2 * (n + 1), 2 * (len(rows) + 1)), squeeze=False)
    axes[0, 0].text(0.5, 0.5, "Original", ha="center", va="center", fontweight="bold")
    for j, name in enumerate(rows):
        caption = f"{name}\n{captions[name]}" if captions and name in captions else name
        axes[j + 1, 0].text(0.5, 0.5, caption, ha="center", va="center", fontsize=9)
    for i in range(n):
        axes[0, i + 1].imshow(_img(originals[i]))
        axes[0, i + 1].set_title(f"label {int(labels[i])}", fontsize=8)
        for j, name in enumerate(rows):
            axes[j + 1, i + 1].imshow(_img(reconstructions[name][i]))
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_curves(x: list[float], series: dict[str, list[float]], xlabel: str, ylabel: str, path: Path) -> Path:
    """One line per series over a shared x axis (sweeps)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, ys in series.items():
        ax.plot(x, ys, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
