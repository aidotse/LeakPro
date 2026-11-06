#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Contract-test cases of object detection: box targets, the detection modality, objectives and evaluators.

The tiny detectors and box targets here are also used by the detection behaviour tests.
"""
from __future__ import annotations

import copy
from functools import lru_cache

import torch
from torch import Tensor, nn

from leakpro.attacks.gia_attacks.modular.components.evaluators import BoxEvaluator, CompositeEvaluator, ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.modalities.detection import DetectionImageModality
from leakpro.attacks.gia_attacks.modular.components.objectives import (
    CriterionDetection,
    HuggingFaceDetection,
    TorchvisionDetection,
)
from leakpro.attacks.gia_attacks.modular.components.targets import JointBoxTarget
from leakpro.attacks.gia_attacks.modular.components.utils.boxes import pad_boxes
from leakpro.attacks.gia_attacks.modular.core.state import State, TargetInfo
from leakpro.tests.gia.contract.registry import (
    EVALUATORS,
    MARKS,
    MODALITIES,
    OBJECTIVES,
    TARGET_SOURCES,
    EvaluatorCase,
    ModalityCase,
    ObjectiveCase,
    TargetSourceCase,
    _needs,
    _psnr_at_identity,
)

DET_SIZE = 64


def _box_target(n: int = 2) -> TargetInfo:
    boxes_list = [torch.tensor([[10.0, 10.0, 30.0, 30.0]]), torch.tensor([[5.0, 5.0, 20.0, 20.0]])][:n]
    labels_list = [torch.tensor([1]), torch.tensor([2])][:n]
    boxes, labels, mask = pad_boxes(boxes_list, labels_list, k_max=1)
    return TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)


def _det_input() -> Tensor:
    return torch.rand(2, 3, DET_SIZE, DET_SIZE, generator=torch.Generator().manual_seed(4))


TARGET_SOURCES.update({
    JointBoxTarget: lambda: TargetSourceCase(
        JointBoxTarget(oracle_labels=_box_target().box_labels, k_max=1), state=State,
        ctx_kwargs={"modality": DetectionImageModality(), "input_shape": (2, 3, DET_SIZE, DET_SIZE)}),
})


@lru_cache(maxsize=None)
def _retinanet() -> nn.Module:
    from torchvision.models.detection import retinanet_resnet50_fpn  # noqa: PLC0415

    torch.manual_seed(0)
    return retinanet_resnet50_fpn(weights=None, weights_backbone=None, min_size=DET_SIZE, max_size=DET_SIZE)


@lru_cache(maxsize=None)
def _detr() -> nn.Module:
    from transformers import DetrConfig, DetrForObjectDetection, ResNetConfig  # noqa: PLC0415

    torch.manual_seed(0)
    backbone = ResNetConfig(embedding_size=8, hidden_sizes=[8, 8, 8, 8], depths=[1, 1, 1, 1], out_features=["stage4"])
    config = DetrConfig(use_timm_backbone=False, use_pretrained_backbone=False, backbone=None, backbone_config=backbone,
                        d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                        decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16, num_queries=4,
                        num_labels=3, attn_implementation="eager")
    return DetrForObjectDetection(config)


def _box_rows(target: TargetInfo) -> Tensor:
    """A stand-in target encoder: the padded boxes, flattened per image."""
    return target.boxes.reshape(target.boxes.shape[0], -1) / DET_SIZE


def _criterion(out: Tensor, encoded: Tensor) -> Tensor:
    """A stand-in detection criterion: per-image squared error between the output and the encoded boxes."""
    return ((out - encoded) ** 2).mean(dim=-1)


OBJECTIVES.update({
    TorchvisionDetection: lambda: ObjectiveCase(
        TorchvisionDetection(), copy.deepcopy(_retinanet()).train(), _det_input(), _box_target()),
    HuggingFaceDetection: lambda: ObjectiveCase(
        HuggingFaceDetection(img_size=DET_SIZE), copy.deepcopy(_detr()).train(), _det_input(), _box_target()),
    CriterionDetection: lambda: ObjectiveCase(
        CriterionDetection(_criterion, _box_rows, sum_output=True),
        nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4)),
        _det_input(), _box_target()),
})

MODALITIES.update({
    DetectionImageModality: lambda: ModalityCase(DetectionImageModality(), _det_input()),
})


def _box_state() -> State:
    target = _box_target()
    gt = {"boxes": target.boxes, "labels": target.box_labels, "mask": target.box_mask}
    return State(target=target, artifacts={BoxEvaluator.GT_KEY: gt})


_BOX_EXPECTED = {"precision_at_iou": 1.0, "recall_at_iou": 1.0, "f1_at_iou": 1.0, "mean_matched_iou": 1.0,
                 "label_accuracy_on_matches": 1.0}

EVALUATORS.update({
    BoxEvaluator: lambda: EvaluatorCase(BoxEvaluator(), _det_input(), _det_input(), state=_box_state(),
                                        expected=_BOX_EXPECTED),
    # Replaces the core image + labels composite: the detection composite of two image-space evaluators.
    CompositeEvaluator: lambda: EvaluatorCase(
        CompositeEvaluator({"image": ImageEvaluator(), "boxes": BoxEvaluator()}), _det_input(), _det_input(),
        state=_box_state(),
        expected={"image.psnr": _psnr_at_identity, "image.ssim": 1.0,
                  **{f"boxes.{k}": v for k, v in _BOX_EXPECTED.items()}}),
})

MARKS.update({
    TorchvisionDetection: (_needs("torchvision"),),
    HuggingFaceDetection: (_needs("transformers"),),
})
