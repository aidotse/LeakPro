import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === paths (edit if needed) ===
FOLDER_NOINV = "gower_plots"
FOLDER_INV   = "gower_plots_with_inv_loss_lr_0002_bz_256_n_1_alpha.2"

CLASSES = list(range(20))   # 0..19
BINS_DEFAULT = np.linspace(0.0, 1.0, 31)  # Gower distance is in [0,1]

def read_class_csv(folder, cls):
    """Try to read either raw distances or pre-binned histogram CSV for a class.

    Returns:
        ("dist", distances_np)  OR  ("hist", (bin_edges_np, counts_np))
    Raises FileNotFoundError if nothing found.

    """
    # Most likely filenames (try in order)
    candidates = [
        f"gower_distances_synClass_{cls}.csv",   # raw distances
        f"gower_hist_data_synClass_{cls}.csv",   # pre-binned
        f"gower_hist_synClass_{cls}.csv",        # user-provided name; could be either
    ]
    # Add a fallback: scan folder for any csv that ends with _{cls}.csv
    for name in candidates + sorted([f for f in os.listdir(folder) if f.endswith(f"_{cls}.csv")]):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        cols = set(df.columns.str.lower())
        # raw distances?
        if "distance" in cols and df["distance"].notna().any():
            return ("dist", df["distance"].to_numpy(dtype=float))
        # pre-binned?
        if {"bin_left", "bin_right", "count"}.issubset(cols):
            # reconstruct bin edges from left/right columns
            bin_left  = df["bin_left"].to_numpy(dtype=float)
            bin_right = df["bin_right"].to_numpy(dtype=float)
            counts    = df["count"].to_numpy(dtype=float)
            # edges are lefts plus the last right
            edges = np.r_[bin_left, bin_right[-1]]
            return ("hist", (edges, counts))
    raise FileNotFoundError(f"No CSV found for class {cls} in {folder}")

def counts_from_either(kind, payload, bins):
    """Return counts for the given bins from either raw distances or pre-binned data."""
    if kind == "dist":
        distances = payload
        c, _ = np.histogram(distances, bins=bins)
        return c.astype(int)
    # kind == "hist"
    edges_existing, counts_existing = payload
    # If edges match bins, just return
    if np.allclose(edges_existing, bins):
        return counts_existing.astype(int)
    # Otherwise, approximate by distributing existing bin counts into the requested bins
    # (simple nearest-edge mapping)
    # Map each existing bin's center to a new bin
    centers = 0.5 * (edges_existing[:-1] + edges_existing[1:])
    idx = np.searchsorted(bins, centers, side="right") - 1
    valid = (idx >= 0) & (idx < len(bins) - 1)
    out = np.zeros(len(bins)-1, dtype=float)
    np.add.at(out, idx[valid], counts_existing[valid])
    return out.astype(int)

# --- figure with 20 subplots (4 x 5) ---
fig, axes = plt.subplots(4, 5, figsize=(20, 14), sharex=True, sharey=True)
axes = axes.ravel()

any_plotted = False

for i, cls in enumerate(CLASSES):
    ax = axes[i]
    try:
        kind_noinv, payload_noinv = read_class_csv(FOLDER_NOINV, cls)
        kind_inv,   payload_inv   = read_class_csv(FOLDER_INV,   cls)
    except FileNotFoundError:
        ax.set_visible(False)
        continue

    # Choose a common binning (prefer pre-binned edges if both sides have them and match; else default)
    bins = BINS_DEFAULT
    if kind_noinv == "hist" and kind_inv == "hist":
        edges_noinv, _ = payload_noinv
        edges_inv, _   = payload_inv
        if np.allclose(edges_noinv, edges_inv):
            bins = edges_noinv

    counts_noinv = counts_from_either(kind_noinv, payload_noinv, bins)
    counts_inv   = counts_from_either(kind_inv,   payload_inv,   bins)

    # bar positions
    width = (bins[1] - bins[0]) * 0.45
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(centers - width/2, counts_noinv, width=width, alpha=0.7, label="no filter on labels")
    ax.bar(centers + width/2, counts_inv,   width=width, alpha=0.7, label="same label")

    ax.set_title(f"class {cls}")
    if i % 5 == 0:
        ax.set_ylabel("count")
    if i // 5 == 3:
        ax.set_xlabel("Gower distance")

    any_plotted = True

# Legend once
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)
fig.suptitle("Nearest-Real Gower Distance — with vs without inversion loss", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = "compare_inv_loss_histograms3.png"
plt.savefig(out_path, dpi=200)
plt.close(fig)

if any_plotted:
    pass
else:
    pass





# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# csv_path = "./gower_plots_without_inv_loss/gower_nn_counts_synClass_vs_realClass.csv"  # <- your file
# out_dir  = os.path.dirname(csv_path) or "."
# out_path = os.path.join(out_dir, "gower_nn_counts_heatmap.png")

# # Load: first column is row labels (synthetic class IDs), header is real class IDs
# df = pd.read_csv(csv_path, index_col=0)  # add sep=';' if needed
# df = df.fillna(0)

# # (Optional) ensure numeric ordering of class IDs
# try:
#     df.index = df.index.astype(int)
#     df.columns = df.columns.astype(int)
#     df = df.sort_index().sort_index(axis=1)
# except Exception:
#     pass

# fig, ax = plt.subplots(figsize=(12, 10))
# im = ax.imshow(df.values, aspect="equal")
# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label("Nearest-real count", rotation=90)

# ax.set_xticks(np.arange(df.shape[1]))
# ax.set_xticklabels(df.columns, rotation=45, ha="right")
# ax.set_yticks(np.arange(df.shape[0]))
# ax.set_yticklabels(df.index)

# ax.set_xlabel("Real class id")
# ax.set_ylabel("Synthetic class id")
# ax.set_title("Counts of nearest REAL class per SYN class")

# # (Optional) annotate each cell with the count
# for i in range(df.shape[0]):
#     for j in range(df.shape[1]):
#         ax.text(j, i, f"{int(df.iat[i, j])}", ha="center", va="center", fontsize=7)

# plt.tight_layout()
# plt.savefig(out_path, dpi=200)
# plt.close(fig)
# print(f"Saved heatmap to: {out_path}")
