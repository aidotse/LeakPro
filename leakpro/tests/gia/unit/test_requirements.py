#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.components.losses.regularization import BNStatsLoss
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.tests.gia import helpers

KW = {"mean": helpers.CIFAR_MEAN, "std": helpers.CIFAR_STD, "device": helpers.DEVICE}


def _run(attack, **kwargs):
    model = helpers.tiny_bn_cnn().train()
    images, labels = helpers.client_batch()
    obs = helpers.observe(attack, model, images, labels)
    return run_attack(attack, model, obs, tuple(images.shape), helpers.DEVICE, target=TargetInfo(value=labels), **kwargs)


def test_missing_bn_access_fails_before_any_optimization() -> None:
    attack = presets.huang(**KW, max_iter=1)
    attack.threat_model = ThreatModel()
    with pytest.raises(ValueError, match="bn_stats.*has_bn_statistics"):
        _run(attack)


def test_proxy_dataloader_reaches_the_bn_loss() -> None:
    attack = presets.huang(**KW, max_iter=1)
    attack.stages[0].losses["bn_stats"] = (BNStatsLoss(strategy="proxy"), 1.0)
    images, labels = helpers.client_batch(batch_size=4, seed=9)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=4)
    seen: list[dict[str, float]] = []

    class _Components(Callback):
        def on_step_end(self, event) -> None:  # noqa: ANN001
            seen.append(dict(event.components))

    state, _ = _run(attack, proxy_dataloader=loader, callbacks=[_Components()])
    assert state.loss == state.loss
    # The proxy statistics are an actual term of the objective: nonzero, and part of the total.
    assert seen[0]["bn_stats"] > 0.0
    assert state.loss > 0.0

    baseline = presets.huang(**KW, max_iter=1)
    baseline.stages[0].losses.pop("bn_stats")
    without, _ = _run(baseline)
    assert state.loss != without.loss
