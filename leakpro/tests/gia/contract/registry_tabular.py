#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Contract-test cases of the tabular attacks: the tabular representation, modalities and evaluator."""
from __future__ import annotations

import pytest

from leakpro.attacks.gia_attacks.modular.components.evaluators import TabularEvaluator
from leakpro.attacks.gia_attacks.modular.components.modalities.tabular import RawTabularModality, TabularModality
from leakpro.attacks.gia_attacks.modular.components.representations.tabular import TabularRepresentation
from leakpro.attacks.gia_attacks.modular.components.tabular.schema import TabularSchema
from leakpro.tests.gia.contract.registry import (
    EVALUATORS,
    MARKS,
    MODALITIES,
    REPRESENTATIONS,
    TABULAR_SHAPE,
    EvaluatorCase,
    ModalityCase,
    RepresentationCase,
    _tabular_rows,
)

TABULAR_SCHEMA = TabularSchema(num_features=TABULAR_SHAPE[1], continuous_cols=[0, 1], categorical_col_groups=[(2, 3)])

REPRESENTATIONS.update({
    TabularRepresentation: lambda: RepresentationCase(
        TabularRepresentation(TABULAR_SCHEMA),
        ctx_kwargs={"modality": TabularModality(TABULAR_SCHEMA), "input_shape": TABULAR_SHAPE},
        image=False, bounded=False),
})

MODALITIES.update({
    TabularModality: lambda: ModalityCase(TabularModality(TABULAR_SCHEMA), _tabular_rows()),
    RawTabularModality: lambda: ModalityCase(RawTabularModality(), _tabular_rows()),
})

EVALUATORS.update({
    TabularEvaluator: lambda: EvaluatorCase(TabularEvaluator(TABULAR_SCHEMA), _tabular_rows(), _tabular_rows(),
                                            expected={"reconstruction_accuracy": 100.0}),
})

MARKS.update({
    TabularModality: (pytest.mark.xfail(
        strict=True,
        reason="By design: the softmax relaxation is not invertible, so strip_model_input returns the "
               "probabilities, not the logits the representation holds (see TabularModality)."),),
})
