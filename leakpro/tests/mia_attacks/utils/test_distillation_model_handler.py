#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Test the distillation model handler module."""

import numpy as np
import torch
from dotmap import DotMap
from torch import Tensor, nn

from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler
from leakpro.tests.input_handler.tabular_input_handler import TabularInputHandler

# The target's optimizer (lr=0.001) is too slow to converge within a test budget,
# so the distillation student gets its own optimizer config.
DISTILLATION_CONFIG = DotMap({"optimizer": {"name": "sgd", "params": {"lr": 0.1, "momentum": 0.9}}})


class ConstantTeacher(nn.Module):
    """Teacher that predicts the same class for every input, regardless of the true label."""

    def __init__(self, num_outputs: int, constant_class: int) -> None:
        super().__init__()
        self.num_outputs = num_outputs
        self.constant_class = constant_class
        # Unused parameter so optimizers/device moves treat this as a normal model
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        logits = torch.full((x.shape[0], self.num_outputs), -5.0, device=x.device)
        if self.num_outputs == 1:
            # Single-logit binary model: constant_class 1 -> positive logit
            logits[:, 0] = 5.0 if self.constant_class == 1 else -5.0
        else:
            logits[:, self.constant_class] = 5.0
        return logits


def _student_predictions(student: nn.Module, handler: ImageInputHandler, indices: np.ndarray) -> np.ndarray:
    student = student.cpu()
    student.eval()
    data_loader = handler.get_dataloader(indices, batch_size=64)
    preds = []
    with torch.no_grad():
        for data, _ in data_loader:
            output = student(data)
            if output.shape[1] == 1:
                preds.append((output[:, 0] > 0).long().numpy())
            else:
                preds.append(output.argmax(dim=1).numpy())
    return np.concatenate(preds)


def test_label_only_distillation_follows_teacher_not_ground_truth(image_handler: ImageInputHandler) -> None:
    """Label-only distillation must learn the teacher's predicted labels, not the dataset labels."""
    image_handler.configs.distillation_model = DISTILLATION_CONFIG
    dm = DistillationModelHandler(image_handler)

    constant_class = 3
    teacher = ConstantTeacher(num_outputs=12, constant_class=constant_class)
    dm.add_student_teacher_pair("label_only_multiclass", teacher)

    indices = np.array(image_handler.test_indices)
    checkpoints = dm.distill_model("label_only_multiclass", 20, indices, label_only=True)
    assert len(checkpoints) == 20

    preds = _student_predictions(checkpoints[-1], image_handler, indices)
    teacher_match = np.mean(preds == constant_class)

    # The fixture spreads ground-truth labels over 12 classes, so a student that
    # (incorrectly) distills on ground truth cannot collapse onto the teacher's class.
    assert teacher_match >= 0.8


def test_label_only_distillation_binary_follows_teacher(tabular_handler: TabularInputHandler) -> None:
    """Label-only distillation on a single-logit model must use the teacher's hard labels."""
    tabular_handler.configs.distillation_model = DISTILLATION_CONFIG
    dm = DistillationModelHandler(tabular_handler)

    teacher = ConstantTeacher(num_outputs=1, constant_class=1)
    dm.add_student_teacher_pair("label_only_binary", teacher)

    indices = np.array(tabular_handler.test_indices)
    checkpoints = dm.distill_model("label_only_binary", 20, indices, label_only=True)

    preds = _student_predictions(checkpoints[-1], tabular_handler, indices)
    assert np.mean(preds == 1) >= 0.8
