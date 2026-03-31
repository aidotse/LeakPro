import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Root directory where all your experiment folders are
ROOT = Path(".")   # change if needed

# Collect all folders that match the pattern
folders = [p for p in ROOT.iterdir() if p.is_dir() and "genlr_" in p.name]

# Sort for consistent order
folders = sorted(folders)

# Number of subplots
n = len(folders)
cols = 5   # number of columns in grid
print(cols)
print(n)
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

# Flatten axes for easy iteration
axes = axes.flatten()

for ax, folder in zip(axes, folders):
    img_path = folder / "losses_combo.png"
    if img_path.exists():
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(folder.name, fontsize=10)
        ax.axis("off")
    else:
        ax.set_visible(False)

# Hide any extra unused axes
for ax in axes[len(folders):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("all_losses_combo2.png", dpi=200, bbox_inches="tight")
plt.show()