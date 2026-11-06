#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: AGPL-3.0-only
#
"""Gradient inversion against the YOLOv8-style detector of this folder, through LeakPro's generic ``detection`` preset.

The YOLO model and its ``ComputeLoss`` live here (derived from https://github.com/jahongir7174/YOLOv8-pt, see LICENSE);
the library only knows detection objectives, so the YOLO specifics are the criterion and ``yolo_head_mode`` below.

The attack recovers the image only: ``ComputeLoss`` assigns its targets under ``torch.no_grad()``, so the box
coordinates get no gradient through this criterion and cannot be optimised. The true boxes are passed to
``run_attack(..., target=...)`` instead, and the demo reports image metrics only.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from torch import Tensor, nn

from leakpro.attacks.gia_attacks.modular.components.objectives import CriterionDetection
from leakpro.attacks.gia_attacks.modular.components.utils.boxes import to_yolo_rows
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.interfaces import ModelMode
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg
from leakpro.attacks.gia_attacks.modular.presets import detection

if TYPE_CHECKING:
    from model import ComputeLoss


def yolo_head_mode(model: nn.Module, mode: ModelMode) -> None:
    """Make the YOLO head return raw per-scale outputs in eval mode (``head.simulate_train_on_eval``)."""
    if hasattr(model, "head"):
        model.head.simulate_train_on_eval = mode == "eval"


def yolo_objective(compute_loss: ComputeLoss, img_size: int, *, sum_output: bool = True) -> CriterionDetection:
    """The YOLO client's loss: ``ComputeLoss`` on rows built from the boxes."""
    rows = partial(to_yolo_rows, img_size=img_size)
    return CriterionDetection(compute_loss, rows, sum_output=sum_output, prepare=yolo_head_mode)


def yolo_attack(
    compute_loss: ComputeLoss,
    img_size: int,
    oracle_labels: Tensor,
    k_max: int,
    *,
    client_epochs: int = 1,
    client_lr: float = 0.1,
    model_mode: ModelMode = "eval",
    **detection_kwargs: Any,  # noqa: ANN401 -- forwarded to ``detection``
) -> Attack:
    """FedAvg gradient inversion against YOLO: eval-mode client, ``client_epochs`` local SGD steps at ``client_lr``.

    ``recover`` defaults to ``"image"`` (pass the boxes as ``run_attack(..., target=...)``): ``ComputeLoss`` gives the
    box coordinates no gradient, so ``recover="both"`` would leave them at their initial values.
    """
    protocol = FedAvg(local_epochs=client_epochs, lr=client_lr)
    detection_kwargs.setdefault("tv_weight", 1e-4)
    detection_kwargs.setdefault("recover", "image")
    return detection(oracle_labels, k_max, objective=yolo_objective(compute_loss, img_size), protocol=protocol,
                     model_mode=model_mode, **detection_kwargs)


def main(argv: list[str] | None = None) -> None:
    """Synthetic-image demo (no COCO download); ``--quick`` is a short CPU smoke run."""
    import argparse
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).parent))  # this script imports its sibling model.py
    from model import ComputeLoss, yolo_v8_n_basicblock

    from leakpro.attacks.gia_attacks.modular.components.utils.boxes import pad_boxes
    from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
    from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "outputs" / "yolo_attack")
    a = p.parse_args(argv)
    torch.manual_seed(a.seed)
    device, size = torch.device(a.device), 64
    iters = 20 if a.quick else 2000

    model = yolo_v8_n_basicblock(num_classes=3).to(device).eval()
    images = torch.rand(1, 3, size, size, device=device)
    boxes, labels, mask = pad_boxes([torch.tensor([[10.0, 10.0, 40.0, 40.0]], device=device)],
                                    [torch.tensor([1], device=device)], k_max=1)
    target = TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)

    attack = yolo_attack(ComputeLoss(model), size, labels, 1, max_iter=iters)
    obs = attack.task.victim_signal(model, images, target, input_shape=tuple(images.shape))
    state, ctx = run_attack(attack, model=model, observation=obs, input_shape=tuple(images.shape), device=device,
                            target=target)
    # The boxes were given, not recovered, so only the image metrics say anything about the attack.
    image_metrics = ctx.evaluator.evaluators["image"].evaluate(state.input, images, ctx=ctx, state=state).metrics
    metrics = {f"image.{k}": v for k, v in image_metrics.items()}

    from examples.gia.common import results  # needs the repository root on the path: run as ``python -m``

    rows: dict[str, dict[str, list[float]]] = {}
    for key, value in metrics.items():  # "image.psnr" -> rows["image"]["psnr"]
        group, _, name = key.partition(".")
        rows.setdefault(group, {})[name] = [float(value)]
    results.save(a.out, rows, {"command": "python -m examples.gia.coco_yolo_1_image.yolo_attack " + " ".join(argv if argv is not None else sys.argv[1:]),
                               "quick": a.quick, "seed": a.seed})
    results.emit(rows)


if __name__ == "__main__":
    main()
