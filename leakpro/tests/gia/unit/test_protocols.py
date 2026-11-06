#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Client protocols (FedSGD, FedAvg) and the Task that runs an objective under one."""
import copy
import types
from collections import OrderedDict

import pytest
import torch
import torch.nn.functional as F

from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.meta import MetaModule, MetaSGD
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.attacks.gia_attacks.modular.core.session import RunSession
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.tests.gia import helpers


def _reference_signal(model, x, target, *, fedavg, epochs=1, batch_size=None, lr=0.01):
    """An independent restatement of the client simulation: autograd for FedSGD, MetaModule + MetaSGD for FedAvg."""
    if not fedavg:
        model.zero_grad()
        loss = F.cross_entropy(model(x), target.value)
        params = [p for p in model.parameters() if p.requires_grad]
        return [g.detach() for g in torch.autograd.grad(loss, params)]
    base_params = dict(model.named_parameters())
    base_buffers = dict(model.named_buffers())
    current = {k: v.detach().clone().requires_grad_(True) for k, v in base_params.items()}
    meta_module, meta_opt = MetaModule(model), MetaSGD(lr=lr)
    n = x.shape[0]
    bs = batch_size or n
    for _ in range(epochs):
        meta_module.buffers = OrderedDict({k: v.clone() for k, v in base_buffers.items()})
        for start in range(0, n, bs):
            idx = torch.arange(start, min(start + bs, n))
            loss = F.cross_entropy(meta_module(x[idx], current), target.value[idx])
            current = meta_opt.step(loss, current, create_graph=False)
    return [(current[k] - base_params[k].detach()).detach() for k in base_params]


CASES = [
    ("fedsgd-train", FedSGD(), dict(fedavg=False), "train"),
    ("fedavg-3ep-bs1", FedAvg(local_epochs=3, batch_size=1, lr=0.05), dict(fedavg=True, epochs=3, batch_size=1, lr=0.05),
     "train"),
    ("fedavg-eval", FedAvg(local_epochs=2), dict(fedavg=True, epochs=2), "eval"),
]


@pytest.mark.parametrize(("name", "protocol", "reference_kw", "mode"), CASES, ids=[c[0] for c in CASES])
def test_protocol_matches_the_reference_simulation(name, protocol, reference_kw, mode) -> None:
    images, labels = helpers.client_batch(batch_size=2)
    x = helpers.image_modality().to_model_input(images)
    target = TargetInfo(value=labels)
    reference_model = helpers.tiny_bn_cnn().train(mode == "train")
    reference = types.SimpleNamespace(gradients=_reference_signal(copy.deepcopy(reference_model), x, target,
                                                                  **reference_kw))
    model = copy.deepcopy(reference_model)
    ours = protocol.observe(model, CrossEntropy(), x, target, create_graph=False, has_epoch_prefix=False)
    assert len(ours.gradients) == len(reference.gradients)
    for a, b in zip(ours.gradients, reference.gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert ours.first_loss.ndim == 0


def test_fedavg_is_frozen_configuration() -> None:
    with pytest.raises(Exception):
        FedAvg().lr = 1.0


def test_create_graph_keeps_the_signal_differentiable_in_the_input() -> None:
    images, labels = helpers.client_batch(batch_size=2)
    x = helpers.image_modality().to_model_input(images).requires_grad_(True)
    target = TargetInfo(value=labels)
    for protocol in (FedSGD(), FedAvg(local_epochs=2)):
        out = protocol.observe(helpers.tiny_bn_cnn().train(), CrossEntropy(), x, target, create_graph=True,
                               has_epoch_prefix=False)
        (grad_x,) = torch.autograd.grad(sum(g.pow(2).sum() for g in out.gradients), x)
        assert grad_x.abs().sum() > 0


def test_task_rejects_a_protocol_its_objective_does_not_support() -> None:
    class FedSGDOnly(CrossEntropy):
        supported_protocols = (FedSGD,)

    with pytest.raises(ValueError, match="FedSGDOnly does not support the FedAvg protocol"):
        Task(FedSGDOnly(), FedAvg())


def test_task_rejects_an_unknown_model_mode() -> None:
    with pytest.raises(ValueError, match="model_mode"):
        Task(CrossEntropy(), model_mode="inference")


# ── RunSession.begin_stage() ──────────────────────────────────────────────────

def _make_ctx(model: torch.nn.Module, input_shape: tuple = (1, 3, 8, 8), num_epochs: int | None = None,
              task: Task | None = None) -> object:
    axes = AxisManager() if num_epochs is None else AxisManager(Axis("epoch", num_epochs))
    return types.SimpleNamespace(model=model, device=torch.device("cpu"), axes=axes, input_shape=input_shape,
                                 task=task if task is not None else Task(CrossEntropy()), session=RunSession())


class TestSessionBeginStage:
    """begin_stage(ctx) must create an isolated execution model and record the pristine params/buffers."""

    def test_simulate_dummy_signal_requires_begin_stage(self):
        ctx = _make_ctx(torch.nn.Linear(4, 2), (1, 4))
        with pytest.raises(RuntimeError, match="begin_stage"):
            ctx.task.simulate_dummy_signal(torch.randn(1, 4), TargetInfo(value=torch.tensor([1])), ctx)

    def test_execution_model_is_a_deepcopy_not_the_original(self):
        model = torch.nn.Linear(4, 2)
        ctx = _make_ctx(model)
        ctx.session.begin_stage(ctx)
        assert ctx.session.execution_model is not model

    def test_execution_model_params_equal_original_on_creation(self):
        model = torch.nn.Linear(4, 2)
        ctx = _make_ctx(model)
        ctx.session.begin_stage(ctx)
        for (n_orig, p_orig), (n_shad, p_shad) in zip(model.named_parameters(),
                                                      ctx.session.execution_model.named_parameters()):
            assert n_orig == n_shad
            assert torch.equal(p_orig.data, p_shad.data)

    def test_execution_model_is_in_train_mode_by_default(self):
        ctx = _make_ctx(torch.nn.Linear(4, 2), task=Task(CrossEntropy(), model_mode="train"))
        ctx.session.begin_stage(ctx)
        assert ctx.session.execution_model.training

    def test_execution_model_is_in_eval_mode_when_requested(self):
        ctx = _make_ctx(torch.nn.Linear(4, 2), task=Task(CrossEntropy(), model_mode="eval"))
        ctx.session.begin_stage(ctx)
        assert not ctx.session.execution_model.training

    def test_original_model_mode_unchanged_after_begin_stage(self):
        model = torch.nn.Linear(4, 2)
        model.eval()
        ctx = _make_ctx(model, task=Task(CrossEntropy(), model_mode="train"))
        ctx.session.begin_stage(ctx)
        assert not model.training

    def test_pristine_params_recorded(self):
        model = torch.nn.Linear(4, 2)
        ctx = _make_ctx(model)
        ctx.session.begin_stage(ctx)
        assert len(ctx.session.pristine_params) > 0
        for name, param in model.named_parameters():
            assert ctx.session.pristine_params[name] is param

    def test_pristine_buffers_recorded(self):
        ctx = _make_ctx(torch.nn.Linear(4, 2))
        ctx.session.begin_stage(ctx)
        assert isinstance(ctx.session.pristine_buffers, dict)

    def test_each_stage_gets_a_fresh_execution_model(self):
        ctx = _make_ctx(torch.nn.Linear(4, 2))
        ctx.session.begin_stage(ctx)
        first = ctx.session.execution_model
        ctx.session.begin_stage(ctx)
        assert ctx.session.execution_model is not first


# ── Epoch prefix: declared, not sniffed from tensor rank ──────────────────────

class TestEpochPrefix:
    """On the attacker path the prefix comes from ``ctx.axes``; on the victim path from ``input_shape``.

    These use a 2-D canonical shape (1, 4) because a rank heuristic hardcoding an image rank of 4
    would mis-classify tabular inputs.
    """

    def test_with_epoch_prefix_no_crash(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        task = Task(CrossEntropy(), FedAvg(local_epochs=2))
        ctx = _make_ctx(model, (1, 4), num_epochs=2, task=task)
        ctx.session.begin_stage(ctx)
        obs = task.simulate_dummy_signal(torch.randn(2, 1, 4), TargetInfo(value=torch.tensor([1])), ctx)
        assert [g.shape for g in obs.gradients] == [p.shape for p in model.parameters()]
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in obs.gradients)

    def test_without_epoch_prefix_no_crash(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        task = Task(CrossEntropy(), FedAvg(local_epochs=2))
        ctx = _make_ctx(model, (1, 4), task=task)
        ctx.session.begin_stage(ctx)
        obs = task.simulate_dummy_signal(torch.randn(1, 4), TargetInfo(value=torch.tensor([1])), ctx)
        assert [g.shape for g in obs.gradients] == [p.shape for p in model.parameters()]
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in obs.gradients)

    def test_rank_alone_does_not_imply_a_prefix(self):
        """A 5-D input with no epoch axis is treated as flat — rank is not the signal."""
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2 * 3 * 4 * 5, 2))
        input_shape = (1, 2, 3, 4, 5)
        task = Task(CrossEntropy(), FedAvg(local_epochs=2))
        ctx = _make_ctx(model, input_shape, task=task)
        ctx.session.begin_stage(ctx)
        obs = task.simulate_dummy_signal(torch.randn(*input_shape), TargetInfo(value=torch.tensor([1])), ctx)
        assert [g.shape for g in obs.gradients] == [p.shape for p in model.parameters()]
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in obs.gradients)

    def test_epoch_prefix_uses_different_data_per_epoch(self):
        """With an epoch prefix each epoch trains on model_input[e]; without it every epoch sees the same batch."""
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        e0, e1 = torch.zeros(1, 4), torch.ones(1, 4)
        target = TargetInfo(value=torch.tensor([0]))
        protocol = FedAvg(local_epochs=2, lr=0.5)
        buffers = dict(model.named_buffers())
        prefix = protocol.observe(model, CrossEntropy(), torch.stack([e0, e1]), target, create_graph=False,
                                  has_epoch_prefix=True,
                                  base_params={k: v.detach().clone() for k, v in model.named_parameters()},
                                  base_buffers=buffers)
        no_prefix = protocol.observe(model, CrossEntropy(), e0, target, create_graph=False, has_epoch_prefix=False,
                                     base_params={k: v.detach().clone() for k, v in model.named_parameters()},
                                     base_buffers=buffers)
        assert any(not torch.allclose(a, b) for a, b in zip(prefix.gradients, no_prefix.gradients))

    def test_without_prefix_the_same_tensor_is_used_every_epoch(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(4, 2)
        task = Task(CrossEntropy(), FedAvg(local_epochs=3, lr=0.05))
        ctx = _make_ctx(model, (2, 4), task=task)
        ctx.session.begin_stage(ctx)
        x, target = torch.randn(2, 4), TargetInfo(value=torch.tensor([0, 1]))
        obs = task.simulate_dummy_signal(x, target, ctx)
        reference = _reference_signal(copy.deepcopy(model), x, target, fedavg=True, epochs=3, lr=0.05)
        assert len(obs.gradients) == len(reference)
        for a, b in zip(obs.gradients, reference):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


class TestFedAvgFrozenParameters:
    def test_a_frozen_layer_is_unchanged_and_the_payload_matches_fedsgd(self):
        images, labels = helpers.client_batch()
        x = helpers.image_modality().to_model_input(images)
        target = TargetInfo(value=labels)
        model = helpers.tiny_bn_cnn().train()
        model[0].weight.requires_grad_(False)
        frozen_before = model[0].weight.detach().clone()
        sgd = FedSGD().observe(copy.deepcopy(model), CrossEntropy(), x, target, create_graph=False, has_epoch_prefix=False)
        avg = FedAvg(local_epochs=2, lr=0.1).observe(model, CrossEntropy(), x, target, create_graph=False,
                                                     has_epoch_prefix=False)
        torch.testing.assert_close(model[0].weight, frozen_before, rtol=0, atol=0)
        assert [g.shape for g in avg.gradients] == [g.shape for g in sgd.gradients]

    def test_no_batches_raises(self):
        class _NoBatches:
            def yield_batches(self, *args, **kwargs):
                return iter(())

        with pytest.raises(ValueError, match="no batches"):
            FedAvg(sampler=_NoBatches()).observe(
                torch.nn.Linear(4, 2), CrossEntropy(), torch.zeros(1, 4), TargetInfo(value=torch.tensor([0])),
                create_graph=False, has_epoch_prefix=False)


class TestFedSGDEpochAxis:
    def test_an_epoch_prefixed_input_is_rejected(self):
        with pytest.raises(ValueError, match="FedSGD"):
            FedSGD().observe(torch.nn.Linear(4, 2), CrossEntropy(), torch.zeros(2, 1, 4),
                             TargetInfo(value=torch.tensor([0])), create_graph=False, has_epoch_prefix=True)


def test_fedsgd_task_rejects_an_epoch_prefixed_input():
    model = torch.nn.Linear(4, 2)
    task = Task(CrossEntropy(), FedSGD())
    with pytest.raises(ValueError, match="FedSGD.*epoch axis"):
        task.victim_signal(model, torch.randn(2, 1, 4), TargetInfo(value=torch.tensor([1])), input_shape=(1, 4))
    ctx = _make_ctx(model, (1, 4), num_epochs=2, task=task)
    ctx.session.begin_stage(ctx)
    with pytest.raises(ValueError, match="FedSGD.*epoch axis"):
        task.simulate_dummy_signal(torch.randn(2, 1, 4), TargetInfo(value=torch.tensor([1])), ctx)
