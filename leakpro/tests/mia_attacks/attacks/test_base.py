#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the reference-model requirements of the BASE attack."""

import numpy as np
import pytest
from pydantic import ValidationError

from leakpro.attacks.mia_attacks.base import AttackBASE
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler


def test_base_offline_allows_uneven_out_model_counts(image_handler:ImageInputHandler) -> None:
    """The offline threshold is a mean over OUT models, so any positive count per point is valid.

    The previous assert demanded exactly num_shadow_models // 2 OUT models, which rejected every
    training_data_fraction other than 0.5 and already failed for an odd num_shadow_models.
    """
    base_obj = AttackBASE(image_handler, {"num_shadow_models": 3, "online": False,
                                          "training_data_fraction": 0.7})

    base_obj._check_out_models_available(np.array([1, 2, 3]))
    base_obj._check_out_models_available(np.array([1, 1, 1]))
    base_obj._check_out_models_available(np.array([3, 3, 3]))


def test_base_offline_rejects_points_without_out_models(image_handler:ImageInputHandler) -> None:
    """A point in every shadow model has no OUT reference and must produce a clear error."""
    base_obj = AttackBASE(image_handler, {"num_shadow_models": 3, "online": False,
                                          "training_data_fraction": 0.7})

    with pytest.raises(ValueError) as excinfo:
        base_obj._check_out_models_available(np.array([2, 0, 1]))

    message = str(excinfo.value)
    assert "at least one OUT shadow model" in message
    assert "1 of 3 points" in message


def test_base_rejects_training_data_fraction_of_one(image_handler:ImageInputHandler) -> None:
    """A fraction of 1 puts every point in every shadow model, leaving no OUT references."""
    with pytest.raises(ValidationError):
        AttackBASE(image_handler, {"num_shadow_models": 2, "online": False,
                                   "training_data_fraction": 1.0})
