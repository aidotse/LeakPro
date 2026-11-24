#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.evaluators import LabelEvaluator
from leakpro.attacks.gia_attacks.modular.core.state import State, TargetInfo

IMAGES = torch.rand(3, 3, 8, 8, generator=torch.Generator().manual_seed(1))


def _state(predicted: list[int], truth: list[int]) -> State:
    return State(target=TargetInfo(value=torch.tensor(predicted)),
                 artifacts={LabelEvaluator.GT_KEY: torch.tensor(truth)})


def test_perfect_recovery() -> None:
    m = LabelEvaluator().evaluate(IMAGES, IMAGES, state=_state([4, 5, 6], [4, 5, 6])).metrics
    assert m == {"accuracy": 1.0, "multiset_accuracy": 1.0}


def test_right_labels_on_the_wrong_images_score_zero_positionally_but_one_as_a_multiset() -> None:
    swapped = IMAGES[[1, 0, 2]]            # the reconstruction holds the images in another order
    m = LabelEvaluator().evaluate(swapped, IMAGES, state=_state([4, 5, 6], [4, 5, 6])).metrics
    # Reconstructed row 0 is truth image 1 (label 5) but carries label 4, and vice versa.
    assert m["accuracy"] == pytest.approx(1 / 3)
    assert m["multiset_accuracy"] == 1.0


def test_partial_multiset_overlap() -> None:
    m = LabelEvaluator().evaluate(IMAGES, IMAGES, state=_state([4, 4, 6], [4, 5, 6])).metrics
    assert m["multiset_accuracy"] == pytest.approx(2 / 3)


def test_missing_ground_truth_gives_an_empty_result() -> None:
    result = LabelEvaluator().evaluate(IMAGES, IMAGES, state=State(target=TargetInfo(value=torch.tensor([1, 2, 3]))))
    assert result.metrics == {}


def test_length_mismatch_names_the_three_lengths() -> None:
    with pytest.raises(ValueError, match="2 predicted, 3 true and 3 reconstruction"):
        LabelEvaluator().evaluate(IMAGES, IMAGES, state=_state([4, 5], [4, 5, 6]))
