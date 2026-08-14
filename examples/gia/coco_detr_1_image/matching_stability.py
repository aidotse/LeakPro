"""Is DETR's bipartite matching stable enough for the reconstruction objective to be optimizable?

DETR's set prediction loss assigns ground truth objects to queries with Hungarian matching, recomputed
under no_grad on every forward pass. The attack optimizes pixels, so the predictions move, so the
assignment can change from one iteration to the next. Each distinct assignment defines a different
loss, and therefore a different gradient the attack is trying to match. If the assignment flips often,
the attack is chasing a target that keeps moving, which would explain a reconstruction loss that
plateaus while the gradient signal itself is demonstrably image specific.

This script measures that directly. It walks a straight line in pixel space from a noise image to the
real client image and reports, at each step, how much the assignment differs from the one at the
previous step and from the one at the real image.
"""
import argparse

import torch
from build import add_data_args, add_model_args, build_loader, build_model, model_tag
from transformers.loss.loss_for_object_detection import HungarianMatcher

from leakpro.utils.seed import seed_everything

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=21, help="Points along the noise-to-image path.")
add_model_args(p)
add_data_args(p)
args = p.parse_args()

seed_everything(1234)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(args).eval().to(dev)
print(f"model={model_tag(args)} img_size={args.img_size}")

client_loader, _, _ = build_loader(args)
image, target = client_loader.dataset[0]
image = image.unsqueeze(0).to(dev)
target = [{k: v.to(dev) for k, v in target.items()}]
num_objects = target[0]["class_labels"].numel()

matcher = HungarianMatcher(class_cost=model.config.class_cost, bbox_cost=model.config.bbox_cost,
                          giou_cost=model.config.giou_cost)
noise = torch.randn_like(image)


def assignment(pixels: torch.Tensor) -> torch.Tensor:
    """Which query each ground truth object is matched to, ordered by object index."""
    with torch.no_grad():
        out = model(pixels)
        query_idx, object_idx = matcher({"logits": out.logits, "pred_boxes": out.pred_boxes}, target)[0]
    return query_idx[object_idx.argsort()].cpu()


real = assignment(image)
print(f"objects={num_objects} queries={model.config.num_queries} assignment_at_real_image={real.tolist()}\n")
print(f"{'alpha':>7}{'assignment':>28}{'changed_vs_prev':>18}{'differs_from_real':>20}")

previous = None
flips = 0
for step in range(args.steps):
    alpha = step / (args.steps - 1)
    current = assignment(noise * (1 - alpha) + image * alpha)
    changed = "-" if previous is None else int((current != previous).sum())
    if previous is not None:
        flips += int((current != previous).sum())
    print(f"{alpha:>7.2f}{str(current.tolist()):>28}{str(changed):>18}"
          f"{int((current != real).sum()):>20}")
    previous = current

print(f"\nTotal object-to-query reassignments along the path: {flips} "
      f"over {args.steps - 1} steps and {num_objects} objects "
      f"({flips / ((args.steps - 1) * num_objects):.1%} of object-steps).")
