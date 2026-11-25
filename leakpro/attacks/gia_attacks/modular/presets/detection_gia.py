#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Gradient inversion attack preset against an object detector."""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.aggregators import MeanAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.evaluators import BoxEvaluator, CompositeEvaluator, ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import BNStatsLoss, TVLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.detection import DetectionImageModality
from leakpro.attacks.gia_attacks.modular.components.objectives import TorchvisionDetection
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets.box import JointBoxTarget
from leakpro.attacks.gia_attacks.modular.components.targets.fixed import FixedTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import ClientProtocol, ModelMode, Objective
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.attacks.gia_attacks.modular.presets._options import GradLoss, grad_loss


def detection(
    oracle_labels: Tensor,
    k_max: int,
    *,
    objective: Objective | None = None,
    protocol: ClientProtocol | None = None,
    model_mode: ModelMode = "train",
    recover: Literal["both", "image"] = "both",
    num_seeds: int = 1,
    lr: float = 0.05,
    max_iter: int = 300,
    loss_type: GradLoss = "cosine",
    tv_weight: float = 0.0,
    bn_weight: float = 0.0,
    bn_strategy: Literal["inferred", "running"] = "inferred",
    bn_momentum: float | None = None,
    iou_threshold: float = 0.5,
) -> Attack:
    """Gradient inversion against an object detector, for any detection objective and client protocol.

    Reconstructs the client's private image batch and box coordinates from a
    single intercepted gradient. Object labels are supplied as an oracle —
    see ``JointBoxTarget`` for why joint label optimization has no gradient
    signal to exploit here — and ``k_max`` (the assumed object count per
    image) is an attacker-chosen capacity, not a recovered quantity.

    Only single-stage detectors work: gradient matching needs to differentiate
    the loss-on-gradients back to the pixels (a double backward), and
    torchvision's ``RoIAlign`` (Faster R-CNN and other two-stage detectors)
    does not support that — see ``components/objectives/detection.py``.

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field. Class labels
        are supplied by the caller.

    References:
        - Geiping, Bauermeister, Droge, Moeller, Inverting Gradients - How easy is it to break privacy in federated
          learning?, NeurIPS 2020, arXiv:2003.14053. The cosine gradient-matching objective.

    Deviations:
        - No published attack is reproduced: this is LeakPro's extension of cosine gradient matching to single-stage
          detectors, with joint box optimisation.
        - Class labels are an oracle and the per-image box count ``k_max`` is chosen by the attacker.

    Args:
        oracle_labels: ``[N, k_max]`` int64 class labels, assumed known.
        k_max: Fixed per-image box capacity the reconstruction optimizes.
        objective: The client's detection loss; ``TorchvisionDetection()`` when ``None``. Use
            ``HuggingFaceDetection`` for DETR or ``CriterionDetection`` for a model scored by a separate criterion.
        protocol: ``FedSGD()`` when ``None``; ``FedAvg(...)`` when the client sends a weight update.
        model_mode: Module mode of the simulated client model (``"train"`` or ``"eval"``).
        recover: ``"both"`` optimizes the image and the box coordinates; ``"image"`` holds the boxes at the
            ``TargetInfo`` passed as ``run_attack(..., target=...)`` and recovers the image only.
        num_seeds: With more than one, the attack runs that many independent reconstructions (a seed axis)
            and averages them.
        lr: Adam learning rate of the reconstruction.
        max_iter: Optimization steps.
        loss_type: ``"cosine"`` (Geiping-style) or ``"l2"`` (DLG-style)
            gradient matching.
        tv_weight: Total-variation regularization weight on the pixels; 0 disables it.
        bn_weight: BatchNorm statistics-matching weight; 0 disables it. Above 0 the threat model grants BN
            statistics (and, for ``"inferred"``, the local hyperparameters), and the observation must carry them.
        bn_strategy: ``"inferred"`` inverts the running-statistics momentum update to recover the batch
            statistics and needs a protocol with one local step; ``"running"`` matches the post-step running
            statistics directly.
        bn_momentum: Momentum for ``"inferred"``; ``None`` reads each layer's own momentum.
        iou_threshold: IoU cutoff for a reconstructed box to count as recovered,
            used by the evaluator's precision/recall/f1.

    Returns:
        Attack — pass to ``run_attack()`` together with the detector and the ``ClientObservation`` from
        ``attack.task.victim_signal()``.

    Example::

        from torchvision.models.detection import retinanet_resnet50_fpn
        from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
        from leakpro.attacks.gia_attacks.modular.presets import detection

        model = retinanet_resnet50_fpn(weights=None, weights_backbone=None, min_size=H, max_size=H)
        model.train()
        atk = detection(oracle_labels=labels, k_max=2)
        obs = atk.task.victim_signal(model, images, target, input_shape=images.shape)
        state, ctx = run_attack(atk, model=model, observation=obs,
                                 input_shape=images.shape, device=device)
        ctx.evaluator.evaluators["boxes"].ground_truth = {"boxes": gt_boxes, "labels": gt_labels, "mask": gt_mask}
        result = ctx.evaluator.evaluate(state.input, images, ctx=ctx, state=state)
        print(result.metrics["image.psnr"], result.metrics["boxes.mean_matched_iou"])

    """
    if recover not in ("both", "image"):
        raise ValueError(f"recover must be 'both' or 'image', got {recover!r}")
    objective = objective if objective is not None else TorchvisionDetection()
    losses: dict = {"grad": (grad_loss(loss_type), 1.0)}
    if tv_weight > 0:
        losses["tv"] = (TVLoss(), tv_weight)
    if bn_weight > 0:
        losses["bn_stats"] = (BNStatsLoss(strategy=bn_strategy, momentum=bn_momentum), bn_weight)
    threat = (ThreatModel(has_bn_statistics=True, has_local_hyperparameters=bn_strategy == "inferred")
              if bn_weight > 0 else ThreatModel())
    seeds = num_seeds > 1
    return Attack(
        threat_model=threat,
        modality=DetectionImageModality(),
        evaluator=CompositeEvaluator({"image": ImageEvaluator(), "boxes": BoxEvaluator(iou_threshold=iou_threshold)}),
        task=Task(objective, protocol if protocol is not None else FedSGD(), model_mode=model_mode),
        axes=AxisManager(Axis("seed", num_seeds)) if seeds else AxisManager(),
        stages=[
            OptimStage(
                losses=losses,
                rep=PixelRepresentation(min_val=0.0, max_val=1.0),
                driver=GradientDriver(lr=lr, max_iter=max_iter, return_best=True, optimizer_cls=torch.optim.Adam),
                target_source=FixedTarget() if recover == "image" else JointBoxTarget(oracle_labels=oracle_labels, k_max=k_max),
                aggregator=MeanAggregator() if seeds else None,
            )
        ],
        target_inference=None,
    )
