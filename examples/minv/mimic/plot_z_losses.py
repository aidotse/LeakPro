import os, math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base = "./z_opt_plots"
classes = list(range(20))
paths = [os.path.join(base, f"loss_curve_class_{c}.png") for c in classes]
paths = [p for p in paths if os.path.isfile(p)]

rows, cols = 4, 5  # 20 panels
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
axes = axes.ravel()

for i, ax in enumerate(axes):
    if i < len(paths):
        img = mpimg.imread(paths[i])
        ax.imshow(img)
        ax.set_title(f"class {classes[i]}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
out_path = os.path.join(base, "loss_curves_grid.png")
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)
