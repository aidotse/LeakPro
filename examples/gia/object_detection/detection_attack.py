#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Gradient inversion against an object detector: reconstruct the client's images and boxes from one update.

The victim is always a random-weight tiny DETR or RetinaNet and the private batch is synthetic (random images with one
box each), so nothing is downloaded. ``--quick`` only shortens the optimisation; the result checks that the pipeline
runs, not that the reconstruction is good.
"""
from __future__ import annotations

import torch

from examples.gia.common import cli, runner
from leakpro.attacks.gia_attacks.modular.components.objectives import HuggingFaceDetection, TorchvisionDetection
from leakpro.attacks.gia_attacks.modular.components.utils.boxes import pad_boxes
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.attacks.gia_attacks.modular.presets import detection


def tiny_detr(num_labels: int) -> torch.nn.Module:
    """A small random-weight DETR built from its config, so no download is needed."""
    from transformers import DetrConfig, DetrForObjectDetection, ResNetConfig  # noqa: PLC0415

    backbone = ResNetConfig(embedding_size=8, hidden_sizes=[8, 8, 8, 8], depths=[1, 1, 1, 1], out_features=["stage4"])
    cfg = DetrConfig(use_timm_backbone=False, use_pretrained_backbone=False, backbone=None, backbone_config=backbone,
                     d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
                     encoder_ffn_dim=16, decoder_ffn_dim=16, num_queries=4, num_labels=num_labels, attn_implementation="eager")
    return DetrForObjectDetection(cfg)


def add_args(p) -> None:
    p.add_argument("--detector", choices=("detr", "retinanet"), default="detr")
    p.add_argument("--protocol", choices=("fedsgd", "fedavg"), default="fedsgd")


def main(argv: list[str] | None = None) -> dict:
    """Attack a synthetic client batch of boxed images and report image PSNR and box-recovery metrics."""
    a = cli.parse(__doc__, argv, extra=add_args)
    device, size, n = cli.device_for(a), 64, a.num_images or 2
    torch.manual_seed(a.seed)
    iters = runner.iters(a, 2000)

    # Synthetic private batch: random images with one box each.
    images = torch.rand(n, 3, size, size, device=device)
    boxes, labels, mask = pad_boxes([torch.tensor([[10.0, 10.0, 40.0, 40.0]], device=device)] * n,
                                    [torch.tensor([1], device=device)] * n, k_max=1)
    target = TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)

    if a.detector == "detr":
        model, objective = tiny_detr(3).to(device).train(), HuggingFaceDetection(img_size=size)
    else:
        from torchvision.models.detection import retinanet_resnet50_fpn  # noqa: PLC0415

        model = retinanet_resnet50_fpn(weights=None, weights_backbone=None, min_size=size, max_size=size).to(device).train()
        objective = TorchvisionDetection()
    protocol = FedAvg(lr=0.01) if a.protocol == "fedavg" else FedSGD()

    attack = detection(labels, 1, objective=objective, protocol=protocol, max_iter=iters)
    obs = attack.task.victim_signal(model, images, target, input_shape=tuple(images.shape))
    state, ctx = run_attack(attack, model=model, observation=obs, input_shape=tuple(images.shape), device=device)
    ctx.evaluator.evaluators["boxes"].ground_truth = target
    metrics = ctx.evaluator.evaluate(state.input, images, ctx=ctx, state=state).metrics

    rows: dict[str, dict[str, list[float]]] = {}
    for key, value in metrics.items():  # "image.psnr" -> rows["image"]["psnr"]
        group, _, name = key.partition(".")
        rows.setdefault(group, {})[name] = [float(value)]
    return runner.finish(a, rows, detector=a.detector, protocol=a.protocol, num_images=n, iterations=iters)


if __name__ == "__main__":
    main()
