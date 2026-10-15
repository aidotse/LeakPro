#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for PETRecipe and the shared DP-SGD training path (CPU, tiny models)."""

import io
import pickle
import typing

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from leakpro.optimization import PETRecipe, fit_dpsgd, make_opacus_compatible, train_with_dpsgd
from leakpro.optimization import training
from leakpro.optimization.training import ACCOUNTANTS, _optimizer_is_stale, _patch_residual_blocks, _rebind_optimizer
from leakpro.schemas import PrivacyUtilityConfig


def _toy_data(n=600, dim=8, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n, dim)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, classes, size=n))
    return {"x": x, "y": y, "train": rng.permutation(n)[:200]}


def _toy_recipe(data, dim=8, classes=3, epochs=1):
    def make_loader(idx, cfg):
        return DataLoader(TensorDataset(data["x"][idx], data["y"][idx]),
                          batch_size=int(cfg["batch_size"]), shuffle=True)

    return PETRecipe(
        make_model=lambda cfg: nn.Linear(dim, classes),
        make_optimizer=lambda params, cfg: optim.SGD(params, lr=cfg["learning_rate"]),
        make_loader=make_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=epochs,
    )


def _batchnorm_recipe(data, dim=8, classes=3, epochs=1):
    """A model Opacus rejects as submitted: BatchNorm mixes samples within a batch."""
    def make_loader(idx, cfg):
        return DataLoader(TensorDataset(data["x"][idx], data["y"][idx]),
                          batch_size=int(cfg["batch_size"]), shuffle=True)

    return PETRecipe(
        make_model=lambda cfg: nn.Sequential(
            nn.Linear(dim, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.Linear(16, classes)),
        make_optimizer=lambda params, cfg: optim.SGD(params, lr=cfg["learning_rate"]),
        make_loader=make_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=epochs,
    )


PRIVATE = {"noise_multiplier": 1.0, "max_grad_norm": 1.0, "learning_rate": 0.1, "batch_size": 64}
NONPRIVATE = {"noise_multiplier": 0.0, "max_grad_norm": 1.0, "learning_rate": 0.1, "batch_size": 64}


class TestTrainWithDPSGD:
    def test_private_path_records_finite_epsilon(self):
        data = _toy_data()
        model = train_with_dpsgd(_toy_recipe(data), PRIVATE, data["train"], "cpu")
        assert np.isfinite(model.dp_accounting["epsilon"])
        assert model.dp_accounting["epsilon"] > 0

    def test_nonprivate_path_is_epsilon_inf(self):
        data = _toy_data()
        model = train_with_dpsgd(_toy_recipe(data), NONPRIVATE, data["train"], "cpu")
        assert model.dp_accounting["epsilon"] == float("inf")
        assert model.dp_accounting["accountant"] is None

    def test_invalid_output_kind_rejected(self):
        with pytest.raises(ValueError):
            PETRecipe(
                make_model=lambda cfg: nn.Linear(2, 2),
                make_optimizer=lambda p, cfg: optim.SGD(p, lr=0.1),
                make_loader=lambda idx, cfg: None,
                criterion=nn.CrossEntropyLoss(),
                epochs=1,
                output_kind="probabilities",
            )

    def test_missing_noise_multiplier_is_rejected(self):
        # Fail closed: defaulting the key would silently train non-privately.
        data = _toy_data()
        bad = {k: v for k, v in PRIVATE.items() if k != "noise_multiplier"}
        with pytest.raises(KeyError, match="noise_multiplier"):
            train_with_dpsgd(_toy_recipe(data), bad, data["train"], "cpu")

    def test_missing_max_grad_norm_is_rejected(self):
        data = _toy_data()
        bad = {k: v for k, v in PRIVATE.items() if k != "max_grad_norm"}
        with pytest.raises(KeyError, match="max_grad_norm"):
            train_with_dpsgd(_toy_recipe(data), bad, data["train"], "cpu")

    def test_returned_model_is_unwrapped_from_opacus(self):
        data = _toy_data()
        model = train_with_dpsgd(_toy_recipe(data), PRIVATE, data["train"], "cpu")
        assert isinstance(model, nn.Linear)
        assert not any(k.startswith("_module.") for k in model.state_dict())

    def test_returned_model_supports_backward(self):
        # Taking ._module left Opacus's hooks registered and grad_sample on every
        # parameter: a later backward raised (IndexError in eval mode, the Poisson
        # grad-accumulation error in train mode). Gradient-based attacks need it.
        data = _toy_data()
        model = train_with_dpsgd(_batchnorm_recipe(data), PRIVATE, data["train"], "cpu")
        assert not any(hasattr(p, "grad_sample") for p in model.parameters())
        assert not any(m._forward_hooks or m._backward_hooks for m in model.modules())
        x, y = data["x"][:16], data["y"][:16]
        for mode in (model.eval, model.train):
            mode()
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            assert all(p.grad is not None for p in model.parameters())

    def test_plain_function_criterion_is_accepted(self):
        data = _toy_data()
        recipe = PETRecipe(
            make_model=lambda cfg: nn.Linear(8, 3),
            make_optimizer=lambda params, cfg: optim.SGD(params, lr=cfg["learning_rate"]),
            make_loader=lambda idx, cfg: DataLoader(TensorDataset(data["x"][idx], data["y"][idx]),
                                                    batch_size=int(cfg["batch_size"]), shuffle=True),
            criterion=F.cross_entropy,
            epochs=1,
        )
        model = train_with_dpsgd(recipe, PRIVATE, data["train"], "cpu")
        assert np.isfinite(model.dp_accounting["epsilon"])

    def test_delta_not_below_one_over_n_warns(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(training.logger, "warning", warnings.append)
        data = _toy_data()  # 200 training records -> 1/n = 5e-3
        train_with_dpsgd(_toy_recipe(data), PRIVATE, data["train"], "cpu", delta=1e-5)
        assert not any("delta" in w for w in warnings)
        train_with_dpsgd(_toy_recipe(data), PRIVATE, data["train"], "cpu", delta=1e-2)
        assert any("delta" in w and "1/n" in w for w in warnings)


class TestFitDPSGD:
    """The handler-shaped entry point: caller-built model, loader, criterion, optimizer."""

    def _pieces(self, data, model):
        loader = DataLoader(TensorDataset(data["x"][data["train"]], data["y"][data["train"]]),
                            batch_size=64, shuffle=True)
        return loader, nn.CrossEntropyLoss(), optim.SGD(model.parameters(), lr=0.1)

    def test_matches_recipe_path_accounting(self):
        data = _toy_data()
        model = nn.Linear(8, 3)
        loader, criterion, optimizer = self._pieces(data, model)
        direct = fit_dpsgd(model, loader, criterion, optimizer, epochs=1,
                           noise_multiplier=1.0, max_grad_norm=1.0, device="cpu")
        via_recipe = train_with_dpsgd(_toy_recipe(data), PRIVATE, data["train"], "cpu")
        # Same data, same noise, same batch size, same epochs: one loop, one epsilon.
        assert direct.dp_accounting["epsilon"] == pytest.approx(via_recipe.dp_accounting["epsilon"], rel=1e-9)

    def test_batchnorm_model_gets_a_rebound_optimizer(self):
        # ModuleValidator.fix returns a copy, so every parameter of the trained
        # model is a new object. An optimizer built beforehand still points at
        # the original's parameters and would step none of the trained model's.
        data = _toy_data()
        model = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.Linear(16, 3))
        loader, criterion, optimizer = self._pieces(data, model)
        stale_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        before = [p.detach().clone() for p in model.parameters()]

        trained = fit_dpsgd(model, loader, criterion, optimizer, epochs=1,
                            noise_multiplier=1.0, max_grad_norm=1.0, device="cpu")

        assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in trained.modules())
        assert _optimizer_is_stale(optimizer, trained), "the original optimizer no longer matches the model"
        assert {id(p) for p in trained.parameters()} != stale_ids
        # Every parameter of the rewritten model moved, GroupNorm affine included.
        for p_before, p_after in zip(before, trained.parameters()):
            if p_before.shape == p_after.shape:
                assert not torch.equal(p_before, p_after)

    def test_compatible_model_keeps_its_optimizer(self):
        data = _toy_data()
        model = nn.Sequential(nn.Linear(8, 16), nn.GroupNorm(4, 16), nn.Linear(16, 3))
        loader, criterion, optimizer = self._pieces(data, model)
        trained = fit_dpsgd(model, loader, criterion, optimizer, epochs=1,
                            noise_multiplier=0.0, max_grad_norm=1.0, device="cpu")
        assert trained is model
        assert not _optimizer_is_stale(optimizer, trained)

    @pytest.mark.parametrize("private", [False, True])
    def test_head_only_optimizer_leaves_the_body_untrained(self, private):
        # An optimizer over the head only is a deliberate choice. Rebinding used
        # to rebuild it over model.parameters(), silently training the body too.
        # Under DP-SGD the body must also have requires_grad=False: Opacus
        # computes per-sample gradients for every trainable parameter and
        # rejects ones the optimizer never consumes. Without DP the optimizer
        # alone decides, which is where the old rebinding bit.
        data = _toy_data()
        model = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 3))
        if private:
            for p in [*model[0].parameters(), *model[1].parameters()]:
                p.requires_grad_(False)
        body_before = model[0].weight.detach().clone()
        head_before = model[3].weight.detach().clone()
        loader, criterion, _ = self._pieces(data, model)
        head_only = optim.SGD(model[3].parameters(), lr=0.1)
        trained = fit_dpsgd(model, loader, criterion, head_only, epochs=1,
                            noise_multiplier=1.0 if private else 0.0, max_grad_norm=1.0, device="cpu")
        assert torch.equal(trained[0].weight, body_before)
        assert not torch.equal(trained[3].weight, head_before)

    def test_nonprivate_leaves_loader_untouched(self):
        data = _toy_data()
        model = nn.Linear(8, 3)
        loader, criterion, optimizer = self._pieces(data, model)
        fit_dpsgd(model, loader, criterion, optimizer, epochs=1,
                  noise_multiplier=0.0, max_grad_norm=1.0, device="cpu")
        assert type(loader) is DataLoader


class TestOptimizerRebinding:
    """_rebind_optimizer: parameters mapped by name, each group keeps its own settings."""

    @staticmethod
    def _model():
        return nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 3))

    def test_each_param_group_keeps_its_settings(self):
        # Flattening the groups made the last group's settings win for every
        # parameter, with no error or warning.
        model = self._model()
        body = [*model[0].parameters(), *model[1].parameters()]
        optimizer = optim.SGD([{"params": body, "lr": 0.1, "weight_decay": 5e-4},
                               {"params": model[3].parameters(), "lr": 0.01, "weight_decay": 0.0}], lr=0.05)
        names_before = {id(p): n for n, p in model.named_parameters()}
        fixed = make_opacus_compatible(model)
        rebound = _rebind_optimizer(optimizer, names_before, fixed)

        assert [(g["lr"], g["weight_decay"]) for g in rebound.param_groups] == [(0.1, 5e-4), (0.01, 0.0)]
        group_of = {id(p): i for i, g in enumerate(rebound.param_groups) for p in g["params"]}
        named = dict(fixed.named_parameters())
        # The GroupNorm that replaced the BatchNorm lands in the BatchNorm's group.
        assert isinstance(fixed[1], nn.GroupNorm)
        assert [group_of[id(named[n])] for n in ("0.weight", "1.weight", "1.bias", "3.weight")] == [0, 0, 0, 1]
        assert not _optimizer_is_stale(rebound, fixed)

    def test_attached_scheduler_does_not_break_rebinding(self):
        # A scheduler adds initial_lr to every group; passing the group settings
        # as constructor kwargs raised TypeError.
        model = self._model()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        optim.lr_scheduler.StepLR(optimizer, step_size=1)
        names_before = {id(p): n for n, p in model.named_parameters()}
        rebound = _rebind_optimizer(optimizer, names_before, make_opacus_compatible(model))
        assert rebound.param_groups[0]["initial_lr"] == 0.1


class TestOpacusCompatibility:
    def test_batchnorm_model_trains_under_dpsgd(self):
        # Before the ModuleValidator pass this raised straight out of make_private.
        data = _toy_data()
        model = train_with_dpsgd(_batchnorm_recipe(data), PRIVATE, data["train"], "cpu")
        assert np.isfinite(model.dp_accounting["epsilon"])

    def test_batchnorm_is_replaced_and_inplace_disabled(self):
        fixed = make_opacus_compatible(
            nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True)))
        assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in fixed.modules())
        assert not any(getattr(m, "inplace", False) for m in fixed.modules())

    def test_nonprivate_anchor_shares_the_private_architecture(self):
        # Every point on a frontier must be the same model. If only private
        # configs got BatchNorm -> GroupNorm, the anchor's utility gap would
        # mix the cost of DP noise with the cost of an architecture change.
        data = _toy_data()
        anchor = train_with_dpsgd(_batchnorm_recipe(data), NONPRIVATE, data["train"], "cpu")
        private = train_with_dpsgd(_batchnorm_recipe(data), PRIVATE, data["train"], "cpu")

        def leaves(model):
            return [type(m).__name__ for m in model.modules() if not list(m.children())]

        assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in anchor.modules())
        assert leaves(anchor) == leaves(private)

    def test_accountant_is_recorded_with_the_epsilon(self):
        data = _toy_data()
        model = train_with_dpsgd(_toy_recipe(data), PRIVATE, data["train"], "cpu", accountant="rdp")
        assert model.dp_accounting["accountant"] == "rdp"

    def test_overridden_residual_forward_is_left_alone(self):
        torchvision = pytest.importorskip("torchvision")

        class CustomResidualBlock(torchvision.models.resnet.BasicBlock):
            """A residual block with its own forward — must survive the patcher untouched."""

            def forward(self, x):  # noqa: ANN001, ANN201, D102
                return self.conv1(x) * 1.0

        # The patcher swaps a stock block's class for an out-of-place subclass,
        # so that is what "was it patched?" means.
        custom = CustomResidualBlock(8, 8)
        stock = torchvision.models.resnet.BasicBlock(8, 8)
        _patch_residual_blocks(nn.Sequential(custom, stock))
        assert type(custom) is CustomResidualBlock      # subclass left untouched
        assert type(stock) is training._OpacusBasicBlock  # stock block patched

    @pytest.mark.parametrize("block", ["basic", "bottleneck"])
    def test_patched_resnet_still_pickles(self, block):
        # The patch used to bind a nested function as an instance forward; such a
        # model failed to unpickle ("'BasicBlock' object has no attribute
        # 'basic_forward'"), breaking whole-model checkpoints and spawned workers.
        torchvision = pytest.importorskip("torchvision")
        model = (torchvision.models.resnet18(num_classes=3) if block == "basic"
                 else nn.Sequential(torchvision.models.resnet.Bottleneck(16, 4)))
        fixed = make_opacus_compatible(model).eval()
        x = torch.randn(2, 3 if block == "basic" else 16, 32, 32)

        restored = pickle.loads(pickle.dumps(fixed))
        buffer = io.BytesIO()
        torch.save(fixed, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=False)
        for copy in (restored, loaded):
            assert torch.equal(copy(x), fixed(x))

    def test_compatible_model_is_returned_unchanged(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.GroupNorm(4, 16))
        assert make_opacus_compatible(model) is model

    def test_resnet_residual_block_trains_under_dpsgd(self):
        # torchvision's `out += identity` is in-place and lives in forward, so
        # ModuleValidator cannot see it; without the patch Opacus's hooks fail here.
        torchvision = pytest.importorskip("torchvision")
        rng = np.random.default_rng(0)
        x = torch.tensor(rng.normal(size=(64, 4, 8, 8)), dtype=torch.float32)
        y = torch.tensor(rng.integers(0, 3, size=64))
        data = {"x": x, "y": y, "train": np.arange(64)}

        recipe = PETRecipe(
            make_model=lambda cfg: nn.Sequential(
                torchvision.models.resnet.BasicBlock(4, 4),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)),
            make_optimizer=lambda params, cfg: optim.SGD(params, lr=cfg["learning_rate"]),
            make_loader=lambda idx, cfg: DataLoader(
                TensorDataset(data["x"][idx], data["y"][idx]), batch_size=int(cfg["batch_size"]), shuffle=True),
            criterion=nn.CrossEntropyLoss(),
            epochs=1,
        )
        model = train_with_dpsgd(recipe, PRIVATE, data["train"], "cpu")
        assert np.isfinite(model.dp_accounting["epsilon"])


class TestAccountant:
    def test_gdp_is_not_offered(self):
        # Opacus's GDP accountant is experimental and can underestimate epsilon.
        assert set(typing.get_args(PrivacyUtilityConfig.model_fields["accountant"].annotation)) == set(ACCOUNTANTS)
        assert "gdp" not in ACCOUNTANTS
        data = _toy_data()
        with pytest.raises(ValueError, match="accountant"):
            train_with_dpsgd(_toy_recipe(data), PRIVATE, data["train"], "cpu", accountant="gdp")

    def test_accountant_is_configurable_and_changes_epsilon(self):
        data = _toy_data()
        recipe = _toy_recipe(data)
        prv = train_with_dpsgd(recipe, PRIVATE, data["train"], "cpu", accountant="prv")
        rdp = train_with_dpsgd(recipe, PRIVATE, data["train"], "cpu", accountant="rdp")
        # PRV is the tighter bound, so it must not report a larger epsilon than RDP.
        assert prv.dp_accounting["epsilon"] < rdp.dp_accounting["epsilon"]

    def test_default_accountant_is_prv(self):
        data = _toy_data()
        recipe = _toy_recipe(data)
        default = train_with_dpsgd(recipe, PRIVATE, data["train"], "cpu")
        prv = train_with_dpsgd(recipe, PRIVATE, data["train"], "cpu", accountant="prv")
        assert default.dp_accounting["accountant"] == "prv"
        assert default.dp_accounting["epsilon"] == pytest.approx(prv.dp_accounting["epsilon"], rel=1e-9)
