#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""BNStatsLoss(strategy="inferred") recovers the client's true batch statistics, and refuses observations it cannot invert.

PyTorch updates ``running_var`` with the unbiased variance over every element of a channel (N*H*W for
BatchNorm2d, N*L for BatchNorm1d on 3-D input, N for 2-D input), so the inversion must undo the Bessel
factor of that count, not of the batch size. At the client's own data the BN term is then zero.
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from leakpro.attacks.gia_attacks.modular.components.losses.regularization import BNStatsLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.interfaces import Modality
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.simulate import simulate_client
from leakpro.attacks.gia_attacks.modular.core.state import Context, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task

CPU = torch.device("cpu")


class _Identity(Modality):
    """Feeds non-image inputs to the model unchanged."""

    def to_model_input(self, data: Tensor) -> Tensor:
        return data

    def strip_model_input(self, model_input: Tensor) -> Tensor:
        return model_input


class _Seq1d(nn.Module):
    """Conv1d -> BatchNorm1d on [N, C, L] -> linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm1d(4)
        self.head = nn.Linear(4 * 8, 5)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.bn(self.conv(x.flatten(2))).flatten(1))


def _bn2d() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 8 * 8, 5))


def _bn1d_flat() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 5))


def _setup(model: nn.Module, shape: tuple[int, ...], protocol=None, mode: str = "train") -> tuple[Context, Tensor]:  # noqa: ANN001
    torch.manual_seed(0)
    model = model.train(mode == "train")
    x = torch.rand(*shape)
    y = torch.arange(shape[0]) % 5
    modality = (ImageNormalizationModality(mean=[0.5] * 3, std=[0.25] * 3, device=CPU) if len(shape) == 4
                else _Identity())
    task = Task(CrossEntropy(), protocol if protocol is not None else FedSGD(), model_mode=mode)
    obs = simulate_client(model, x, TargetInfo(value=y), task=task, modality=modality)
    ctx = Context(model=model, observation=obs, modality=modality, task=task, input_shape=tuple(x.shape), device=CPU)
    ctx.session.begin_stage(ctx)
    return ctx, modality.to_model_input(x)


@pytest.mark.parametrize(
    ("model_fn", "shape"),
    [(_bn2d, (4, 3, 8, 8)), (_Seq1d, (4, 3, 8)), (_bn1d_flat, (4, 3, 8))],
    ids=["BatchNorm2d", "BatchNorm1d-3d", "BatchNorm1d-2d"],
)
def test_inferred_term_is_zero_at_the_clients_batch(model_fn, shape: tuple[int, ...]) -> None:  # noqa: ANN001
    ctx, client_input = _setup(model_fn(), shape)
    bound = BNStatsLoss().bind(ctx)
    try:
        with torch.no_grad():
            ctx.session.execution_model(client_input)
        (hook,) = bound.hooks
        torch.testing.assert_close(hook.var, hook.target_batch_var, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(hook.mean, hook.target_batch_mean, atol=1e-5, rtol=1e-4)
        assert float(hook.r_feature) < 1e-4
    finally:
        bound.release()


def test_inferred_raises_for_fedavg_with_more_than_one_local_step() -> None:
    ctx, _ = _setup(_bn2d(), (4, 3, 8, 8), protocol=FedAvg(local_epochs=3, lr=0.01))
    with pytest.raises(ValueError, match="local step"):
        BNStatsLoss().bind(ctx)


def test_inferred_raises_for_fedavg_with_minibatches() -> None:
    ctx, _ = _setup(_bn2d(), (4, 3, 8, 8), protocol=FedAvg(local_epochs=1, batch_size=2, lr=0.01))
    with pytest.raises(ValueError, match="local step"):
        BNStatsLoss().bind(ctx)


def test_inferred_accepts_single_step_fedavg_and_recovers_the_batch() -> None:
    ctx, client_input = _setup(_bn2d(), (4, 3, 8, 8), protocol=FedAvg(local_epochs=1, lr=0.01))
    bound = BNStatsLoss().bind(ctx)
    try:
        with torch.no_grad():
            ctx.session.execution_model(client_input)
        assert float(bound.hooks[0].r_feature) < 1e-4
    finally:
        bound.release()


def test_inferred_raises_for_an_eval_mode_client() -> None:
    ctx, _ = _setup(_bn2d(), (4, 3, 8, 8), mode="eval")
    with pytest.raises(ValueError, match="train mode"):
        BNStatsLoss().bind(ctx)
