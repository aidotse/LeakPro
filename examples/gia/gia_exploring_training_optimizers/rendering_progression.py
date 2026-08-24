#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Render a video of the GIA reconstruction progression.

Reads the per-improvement reconstruction snapshots saved by main.py
(`reconstruction_series_*.pt` inside an outputs/<run> folder) and turns them into a video,
one frame per improvement, annotated with iteration / loss / PSNR / SSIM.

Usage:
    python rendering_progression.py run_20260819_150834          # a run folder under outputs/
    python rendering_progression.py outputs/run_.../series.pt    # a specific series file
    python rendering_progression.py run_... --fps 8

Writes `progression_<tag>.mp4` (or .gif if ffmpeg is unavailable) next to each series file.
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib import animation

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def find_series_files(arg: str) -> list[Path]:
    """Resolve the CLI argument to a list of reconstruction_series_*.pt files."""
    path = Path(arg)
    candidates = [path, OUTPUTS_DIR / arg]  # allow a full/relative path or a bare run-folder name
    for cand in candidates:
        if cand.is_file():
            return [cand]
        if cand.is_dir():
            files = sorted(cand.glob("reconstruction_series_*.pt"))
            if not files:
                raise FileNotFoundError(f"No reconstruction_series_*.pt found in {cand}")
            return files
    raise FileNotFoundError(f"Could not resolve '{arg}' to a file or a folder under {OUTPUTS_DIR}")


def make_denorm(data: dict):
    """Build a denormalization fn mapping a [*, C, H, W] tensor to displayable [*, H, W, C] in [0, 1].

    Uses saved data_mean/data_std if present (so originals and reconstructions share one color
    space); otherwise a global min-max scaling derived from the reconstructions.
    """
    if "data_mean" in data and "data_std" in data:
        mean, std = data["data_mean"].float(), data["data_std"].float()

        def denorm(x: torch.Tensor) -> torch.Tensor:
            c = x.shape[-3]
            m = mean.reshape(*([1] * (x.dim() - 3)), c, 1, 1)
            s = std.reshape(*([1] * (x.dim() - 3)), c, 1, 1)
            return (x.float() * s + m).clamp(0.0, 1.0).movedim(-3, -1)
    else:
        recons = data["reconstructions"].float()
        lo, hi = recons.amin(), recons.amax()

        def denorm(x: torch.Tensor) -> torch.Tensor:
            return ((x.float() - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0).movedim(-3, -1)

    return denorm


def _show(ax, frame, grayscale: bool):
    """imshow a single [H, W, C] (or [H, W] if grayscale) frame."""
    data = frame[:, :, 0] if grayscale else frame
    return ax.imshow(data, cmap="gray" if grayscale else None, vmin=0.0, vmax=1.0)


def render(series_path: Path, fps: int) -> Path:
    """Render one series file to a video and return the output path."""
    data = torch.load(series_path, map_location="cpu")
    denorm = make_denorm(data)
    recon = denorm(data["reconstructions"])                   # [S, N, H, W, C]
    s_count, n_images, height, width, channels = recon.shape
    iters = data["iterations"].tolist()
    losses = data["losses"].tolist()
    psnr = data["psnr"].tolist()
    ssim = data["ssim"].tolist()

    tag = series_path.stem.replace("reconstruction_series_", "")
    grayscale = channels == 1
    has_orig = "originals" in data
    originals = denorm(data["originals"]) if has_orig else None  # [N, H, W, C]

    n_rows = 2 if has_orig else 1
    recon_row = 1 if has_orig else 0
    fig, axes = plt.subplots(n_rows, n_images, figsize=(2.4 * n_images, 2.8 * n_rows), squeeze=False)

    if has_orig:  # static top row: ground-truth originals
        for i in range(n_images):
            _show(axes[0][i], originals[i], grayscale)
            axes[0][i].set_title(f"original {i}", fontsize=8)
            axes[0][i].axis("off")

    images = []  # animated bottom row: reconstruction
    for i in range(n_images):
        ax = axes[recon_row][i]
        images.append(_show(ax, recon[0, i], grayscale))
        ax.set_title(f"recon {i}", fontsize=8)
        ax.axis("off")

    def title_for(s: int) -> str:
        return (f"{tag}  |  snapshot {s + 1}/{s_count}  |  iter {iters[s]}  |  "
                f"loss {losses[s]:.3f}  |  PSNR {psnr[s]:.2f} dB  |  SSIM {ssim[s]:.3f}")

    suptitle = fig.suptitle(title_for(0), fontsize=9)

    def update(s: int):
        for i in range(n_images):
            frame = recon[s, i]
            images[i].set_data(frame[:, :, 0] if grayscale else frame)
        suptitle.set_text(title_for(s))
        return [*images, suptitle]

    anim = animation.FuncAnimation(fig, update, frames=s_count, interval=1000 / fps, blit=False)

    if animation.writers.is_available("ffmpeg"):
        out_path = series_path.with_name(f"progression_{tag}.mp4")
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
    else:
        out_path = series_path.with_name(f"progression_{tag}.gif")
        print("ffmpeg not available - falling back to GIF.")
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a video of the GIA reconstruction progression.")
    parser.add_argument("run", help="Run folder name under outputs/ (e.g. run_20260819_150834), "
                                     "a folder path, or a reconstruction_series_*.pt file.")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second (default: 5).")
    args = parser.parse_args()

    for series_path in find_series_files(args.run):
        print(f"Rendering {series_path.name} ...")
        out_path = render(series_path, args.fps)
        print(f"Saved video to: {out_path}")


if __name__ == "__main__":
    main()
