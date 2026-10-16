#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""LossTrajectory must score the target's own train/test points (#476).

The shared fixture uses ``train_indices = 0..n`` so audit position and population
index coincide, which hides the bug. The test keeps every other point, reversed, so
positions and population indices differ and points are left over for the shadow model.
"""

from unittest.mock import MagicMock

import numpy as np

from leakpro.attacks.mia_attacks import loss_trajectory
from leakpro.attacks.mia_attacks.loss_trajectory import AttackLossTrajectory
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler


def test_mia_test_data_is_the_targets_train_and_test_points(image_handler: ImageInputHandler,
                                                             monkeypatch: object) -> None:
    """The MIA test set is train + test in population indices, with members labelled by the target's split."""
    train = np.asarray(image_handler.train_indices)[::-2].copy()
    test = np.asarray(image_handler.test_indices)[::-2].copy()
    image_handler.train_indices = train
    image_handler.test_indices = test

    # Shadow training and distillation are irrelevant here; only the data handed to the MIA classifier is checked.
    monkeypatch.setattr(loss_trajectory, "ShadowModelHandler", MagicMock())
    monkeypatch.setattr(loss_trajectory, "DistillationModelHandler", MagicMock())
    loss_trajectory.ShadowModelHandler.return_value.get_shadow_models.return_value = ([MagicMock()], None)

    attack = AttackLossTrajectory(image_handler, {})
    calls = []
    monkeypatch.setattr(attack, "prepare_mia_data",
                        lambda indices, mask, **kwargs: calls.append((indices, mask, kwargs["train_mode"])))
    attack.prepare_attack()

    test_calls = [(indices, mask) for indices, mask, train_mode in calls if not train_mode]
    assert len(test_calls) == 1
    indices, mask = test_calls[0]
    np.testing.assert_array_equal(indices, np.concatenate((train, test)))
    train_set = set(train.tolist())
    assert [int(i) in train_set for i in indices] == mask.tolist()
