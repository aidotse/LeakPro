#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Padded ragged detection targets (components/utils/boxes.py).

A batch has a variable number of boxes per image. TargetInfo.map only ever
transforms Tensor fields, passing anything else through untouched — so a raw
list[Tensor] of per-image boxes would survive index_axis *unsliced*, silently
pairing a minibatch of images with the full batch's boxes. pad_boxes/unpad_boxes
sidestep that by converting to a fixed [N, k_max, ...] tensor + mask up front,
which the existing per-sample-dim slicing already handles correctly — these
tests exist to pin that down, not to test new slicing logic.
"""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.utils.boxes import (
    assert_box_target_valid,
    pad_boxes,
    unpad_boxes,
)
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo


class TestPadBoxes:
    def test_pads_to_k_max_and_masks_correctly(self):
        boxes_list = [
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]),
        ]
        labels_list = [torch.tensor([3]), torch.tensor([0, 5])]

        boxes, labels, mask = pad_boxes(boxes_list, labels_list, k_max=3)

        assert boxes.shape == (2, 3, 4)
        assert labels.shape == (2, 3)
        assert mask.shape == (2, 3)
        assert mask.tolist() == [[True, False, False], [True, True, False]]

    def test_zero_box_image_is_all_padding(self):
        boxes_list = [torch.zeros((0, 4)), torch.tensor([[0.0, 0.0, 1.0, 1.0]])]
        labels_list = [torch.zeros((0,), dtype=torch.long), torch.tensor([1])]

        boxes, labels, mask = pad_boxes(boxes_list, labels_list, k_max=2)

        assert not mask[0].any()
        assert mask[1].tolist() == [True, False]

    def test_mismatched_list_lengths_raises(self):
        with pytest.raises(ValueError):
            pad_boxes([torch.zeros((1, 4))], [], k_max=2)

    def test_overflow_raises_rather_than_truncating(self):
        boxes_list = [torch.zeros((2, 4))]
        labels_list = [torch.zeros((2,), dtype=torch.long)]
        with pytest.raises(ValueError):
            pad_boxes(boxes_list, labels_list, k_max=1)


class TestUnpadBoxes:
    def test_round_trips_through_pad_boxes(self):
        boxes_list = [
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]),
        ]
        labels_list = [torch.tensor([3]), torch.tensor([0, 5])]

        boxes, labels, mask = pad_boxes(boxes_list, labels_list, k_max=4)
        recovered_boxes, recovered_labels = unpad_boxes(boxes, labels, mask)

        for orig, rec in zip(boxes_list, recovered_boxes):
            assert torch.equal(orig, rec)
        for orig, rec in zip(labels_list, recovered_labels):
            assert torch.equal(orig, rec)


class TestPaddedBoxesSurviveTargetInfoSlicing:
    """The whole point of padding: index_axis must not desync boxes from images."""

    def _sample_batch(self):
        boxes_list = [
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]),
        ]
        labels_list = [torch.tensor([3]), torch.tensor([0, 5])]
        return pad_boxes(boxes_list, labels_list, k_max=3)

    def test_batch_dim_slice_keeps_boxes_and_mask_aligned(self):
        boxes, labels, mask = self._sample_batch()
        target = TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)

        sliced = target.index_axis(0, torch.tensor([1]))

        assert torch.equal(sliced.boxes[0], boxes[1])
        assert torch.equal(sliced.box_mask[0], mask[1])
        assert torch.equal(sliced.box_labels[0], labels[1])

    def test_epoch_prefix_slice_keeps_boxes_aligned(self):
        boxes, _labels, mask = self._sample_batch()
        # [E=2, N, K, 4] — epoch 1 is a different set of boxes from epoch 0.
        epoch_boxes = torch.stack([boxes, boxes * 2])
        epoch_mask = torch.stack([mask, mask])
        target = TargetInfo(boxes=epoch_boxes, box_mask=epoch_mask, axis_ndim=1)

        epoch0 = target.index_axis(0, 0)

        assert torch.equal(epoch0.boxes, boxes)
        assert epoch0.axis_ndim == 0


class TestAssertBoxTargetValid:
    def test_boxes_without_mask_raises(self):
        target = TargetInfo(boxes=torch.zeros(2, 3, 4))
        with pytest.raises(ValueError, match="box_mask"):
            assert_box_target_valid(target)

    def test_boxes_with_mask_passes(self):
        target = TargetInfo(boxes=torch.zeros(2, 3, 4), box_mask=torch.zeros(2, 3, dtype=torch.bool))
        assert_box_target_valid(target) is None

    def test_no_boxes_is_a_no_op(self):
        assert_box_target_valid(TargetInfo(value=torch.zeros(1))) is None

    def test_mismatched_mask_shape_raises(self):
        target = TargetInfo(boxes=torch.zeros(2, 3, 4), box_mask=torch.zeros(2, 2, dtype=torch.bool))
        with pytest.raises(ValueError, match="shape"):
            assert_box_target_valid(target)


def test_to_torchvision_targets_applies_the_mask() -> None:
    from leakpro.attacks.gia_attacks.modular.components.utils.boxes import to_torchvision_targets

    boxes, labels, mask = pad_boxes(
        [torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.zeros((0, 4))],
        [torch.tensor([5]), torch.zeros((0,), dtype=torch.long)], k_max=2,
    )
    out = to_torchvision_targets(TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask))
    assert [tuple(d["boxes"].shape) for d in out] == [(1, 4), (0, 4)]
    assert out[0]["labels"].dtype == torch.long and out[0]["labels"].tolist() == [5]


def test_to_detr_labels_normalizes_to_cxcywh() -> None:
    from leakpro.attacks.gia_attacks.modular.components.utils.boxes import to_detr_labels

    boxes, labels, mask = pad_boxes([torch.tensor([[0.0, 0.0, 32.0, 64.0]])], [torch.tensor([2])], k_max=1)
    (d,) = to_detr_labels(TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask), img_size=64)
    assert torch.allclose(d["boxes"], torch.tensor([[0.25, 0.5, 0.5, 1.0]]))
    assert d["class_labels"].tolist() == [2]
