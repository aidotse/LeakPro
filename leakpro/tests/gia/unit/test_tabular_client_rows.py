#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The simulated tabular client trains on its real one-hot rows; only the attacker's logits are softmaxed."""
from __future__ import annotations

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.modalities.tabular import TabularModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.tabular.schema import TabularSchema
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD
from leakpro.attacks.gia_attacks.modular.core.simulate import simulate_client
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task

SCHEMA = TabularSchema(num_features=5, continuous_cols=[0, 1], categorical_col_groups=[(2, 3)])


def _rows() -> torch.Tensor:
    gen = torch.Generator().manual_seed(3)
    rows = torch.randn(3, 5, generator=gen)
    rows[:, 2:5] = nn.functional.one_hot(torch.tensor([0, 2, 1]), 3).float()
    return rows


def test_observation_is_the_gradient_on_the_raw_one_hot_rows() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 2))
    rows, labels = _rows(), torch.tensor([0, 1, 1])
    task = Task(CrossEntropy(), FedSGD(), model_mode="eval")

    obs = simulate_client(model, rows, TargetInfo(value=labels), task=task, modality=TabularModality(SCHEMA))

    expected = torch.autograd.grad(nn.functional.cross_entropy(model(rows), labels), list(model.parameters()))
    for got, want in zip(obs.gradients, expected):
        torch.testing.assert_close(got, want)


def test_client_rows_pass_unchanged_while_the_reconstruction_is_relaxed() -> None:
    modality = TabularModality(SCHEMA)
    rows = _rows()
    assert torch.equal(modality.client_to_model_input(rows), rows)
    relaxed = modality.to_model_input(rows)
    torch.testing.assert_close(relaxed[:, 2:5], torch.softmax(rows[:, 2:5], dim=-1))
