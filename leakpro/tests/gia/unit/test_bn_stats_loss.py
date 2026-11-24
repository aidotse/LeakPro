#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""BNStatsLoss fails fast on missing inputs and pairs observed statistics with the BatchNorm layers they came from."""
from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.losses.regularization import BNStatsLoss
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs
from leakpro.fl_utils.model_utils import InferredBNFeatureHook
from leakpro.tests.gia import helpers
from leakpro.tests.gia.contract.conftest import client_target, make_ctx


def _bind(term: BNStatsLoss, ctx) -> BNStatsLoss:  # noqa: ANN001
    ctx.session.begin_stage(ctx)
    return term.bind(ctx)


def _mixed_norm_model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.LayerNorm([4, 8, 8]),
        nn.BatchNorm2d(4),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, helpers.NUM_CLASSES),
    ).train()


def test_inferred_declares_pre_training_statistics_and_batch_size() -> None:
    assert BNStatsLoss().requirements() == {"bn_stats", "pre_bn_stats", "batch_size"}
    assert BNStatsLoss(strategy="running").requirements() == {"bn_stats"}


@pytest.mark.parametrize("missing", ["pre_bn_stats", "batch_size"])
def test_inferred_bind_raises_without_its_inputs(missing: str) -> None:
    ctx = make_ctx()
    ctx = dataclasses.replace(ctx, observation=dataclasses.replace(ctx.observation, **{missing: None}))
    with pytest.raises(ValueError, match=missing):
        _bind(BNStatsLoss(), ctx)


def test_cumulative_momentum_raises_unless_given_explicitly() -> None:
    model = helpers.tiny_bn_cnn().train()
    model[1].momentum = None
    ctx = make_ctx(model=model)
    with pytest.raises(ValueError, match="momentum"):
        _bind(BNStatsLoss(), ctx)
    bound = _bind(BNStatsLoss(momentum=0.1), ctx)
    assert len(bound.hooks) == 1
    bound.release()


def test_statistics_land_on_the_batchnorm_layer_in_a_mixed_norm_model() -> None:
    ctx = make_ctx(model=_mixed_norm_model())
    assert len(ctx.observation.bn_stats) == 1
    bound = _bind(BNStatsLoss(strategy="running"), ctx)
    try:
        assert len(bound.hooks) == 1
        assert isinstance(bound.hooks[0], InferredBNFeatureHook)
        assert torch.equal(bound.hooks[0].target_batch_mean, ctx.observation.bn_stats[0][0])
    finally:
        bound.release()


def test_layer_count_mismatch_names_both_counts() -> None:
    ctx = make_ctx()
    extra = ctx.observation.bn_stats * 2
    ctx = dataclasses.replace(ctx, observation=dataclasses.replace(ctx.observation, bn_stats=extra, pre_bn_stats=extra))
    with pytest.raises(ValueError, match=r"2.*1|1.*2"):
        _bind(BNStatsLoss(), ctx)


def test_a_model_without_batchnorm_layers_raises_instead_of_contributing_zero() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, helpers.NUM_CLASSES)).train()
    ctx = make_ctx(model=model)
    ctx = dataclasses.replace(ctx, observation=dataclasses.replace(ctx.observation, bn_stats=[], pre_bn_stats=[]))
    bound = _bind(BNStatsLoss(), ctx)
    inputs = LossInputs(recon=torch.zeros(2, 3, 8, 8), target=client_target(), target_obs=ctx.observation, ctx=ctx,
                        dummy_obs=None)
    with pytest.raises(ValueError, match="BatchNorm"):
        bound.compute(inputs)


def _batches(n: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(3)
    return [(torch.randn(4, 3, 8, 8, generator=gen) * (i + 1), torch.randint(0, helpers.NUM_CLASSES, (4,), generator=gen))
            for i in range(n)]


def test_proxy_statistics_are_the_mean_over_all_batches() -> None:
    model = _mixed_norm_model()
    batches = _batches()
    term = BNStatsLoss(strategy="proxy", proxy_lr=0.0)  # lr 0: the weights stay put, so the reference is a plain forward

    stats = term._collect_proxy_stats(model, batches, torch.device("cpu"))

    conv, ln, bn = model[0], model[1], model[2]
    per_batch = []
    for x, _ in batches:
        h = conv(x)
        ln_in, bn_in = h, ln(h)
        per_batch.append((ln_in.mean(), ln_in.var(unbiased=False),
                          bn_in.mean(dim=(0, 2, 3)), bn_in.var(dim=(0, 2, 3), unbiased=False)))
    assert len(stats) == 2  # LayerNorm, BatchNorm2d: every layer, in module order
    torch.testing.assert_close(stats[0][0], torch.stack([b[0] for b in per_batch]).mean(0))
    torch.testing.assert_close(stats[0][1], torch.stack([b[1] for b in per_batch]).mean(0))
    torch.testing.assert_close(stats[1][0], torch.stack([b[2] for b in per_batch]).mean(0))
    torch.testing.assert_close(stats[1][1], torch.stack([b[3] for b in per_batch]).mean(0))


def test_proxy_collection_leaves_the_model_untouched() -> None:
    model = _mixed_norm_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    BNStatsLoss(strategy="proxy", proxy_lr=0.1)._collect_proxy_stats(model, _batches(), torch.device("cpu"))
    after = model.state_dict()
    assert before.keys() == after.keys()
    for k in before:
        torch.testing.assert_close(after[k], before[k], rtol=0, atol=0)
    assert model.training


def test_proxy_collection_raises_when_a_layer_is_never_reached() -> None:
    class Skips(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used, self.unused = nn.BatchNorm2d(3), nn.BatchNorm2d(3)
            self.head = nn.Linear(3, helpers.NUM_CLASSES)

        def forward(self, x):  # noqa: ANN001, ANN201
            return self.head(self.used(x).mean(dim=(2, 3)))

    with pytest.raises(ValueError, match="never reached"):
        BNStatsLoss(strategy="proxy")._collect_proxy_stats(Skips().train(), _batches(1), torch.device("cpu"))
