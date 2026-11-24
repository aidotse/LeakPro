#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import copy

import torch

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg
from leakpro.attacks.gia_attacks.modular.core.simulate import simulate_client
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.tests.gia import helpers

KW = {"mean": helpers.CIFAR_MEAN, "std": helpers.CIFAR_STD, "device": helpers.DEVICE}


def test_fills_every_field_and_leaves_the_model_alone() -> None:
    attack = presets.inverting_gradients(**KW, model_mode="train")
    model = helpers.tiny_bn_cnn().train()
    before = copy.deepcopy(model.state_dict())
    images, labels = helpers.client_batch()
    obs = simulate_client(model, images, TargetInfo(value=labels), task=attack.task, modality=attack.modality)
    assert model.training
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k])
    assert len(obs.gradients) == len(list(model.parameters()))
    torch.testing.assert_close(obs.pre_bn_stats[0][0], before["1.running_mean"])
    assert not torch.allclose(obs.bn_stats[0][0], obs.pre_bn_stats[0][0])
    assert obs.batch_size == 2 and isinstance(obs.loss_value, float)


def test_b22_pre_bn_stats_are_stripped_without_bn_access() -> None:
    obs = ClientObservation(gradients=[torch.ones(1)], pre_bn_stats=[(torch.zeros(1), torch.ones(1))])
    assert ThreatModel().sanitize(obs).pre_bn_stats is None


def test_simulate_client_leaves_the_callers_bn_statistics_and_mode_untouched() -> None:
    model = helpers.tiny_bn_cnn().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    images, labels = helpers.client_batch()
    obs = simulate_client(model, images, TargetInfo(value=labels), task=Task(CrossEntropy()),
                          modality=helpers.image_modality())
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)
    assert not torch.equal(obs.bn_stats[0][0], obs.pre_bn_stats[0][0])


def test_fedavg_observation_carries_the_client_trained_bn_statistics() -> None:
    """With FedAvg the post-training BN statistics are those of a client that really ran the local steps in train mode."""
    epochs, batch_size, lr = 2, 1, 0.05
    task = Task(CrossEntropy(), FedAvg(local_epochs=epochs, batch_size=batch_size, lr=lr), model_mode="train")
    model = helpers.tiny_bn_cnn().train()
    images, labels = helpers.client_batch()
    modality = helpers.image_modality()
    obs = simulate_client(model, images, TargetInfo(value=labels), task=task, modality=modality)

    client = copy.deepcopy(model).train()
    optimizer = torch.optim.SGD(client.parameters(), lr=lr)
    x = modality.to_model_input(images)
    for _ in range(epochs):
        for i in range(0, len(x), batch_size):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(client(x[i:i + batch_size]), labels[i:i + batch_size]).backward()
            optimizer.step()
    expected = [(m.running_mean, m.running_var) for m in client.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert not torch.equal(obs.bn_stats[0][0], obs.pre_bn_stats[0][0])
    for (mean, var), (want_mean, want_var) in zip(obs.bn_stats, expected):
        torch.testing.assert_close(mean, want_mean)
        torch.testing.assert_close(var, want_var)
