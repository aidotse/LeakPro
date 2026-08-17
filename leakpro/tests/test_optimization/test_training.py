#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for PETRecipe and the shared DP-SGD training path (CPU, tiny models)."""

import numpy as np
import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from leakpro.optimization import (
    AttackScores,
    PETRecipe,
    build_campaign_fns,
    confidence_signal,
    make_opacus_compatible,
    train_with_dpsgd,
)


def _toy_splits(n=600, dim=8, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n, dim)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, classes, size=n))
    order = rng.permutation(n)
    return {
        "x": x, "y": y,
        "target_train": order[:200],
        "ref_pool": order[200:400],
        "audit_members": order[:100],
        "audit_nonmembers": order[400:500],
        "utility_eval": order[500:],
    }


def _toy_recipe(splits=None, dim=8, classes=3, epochs=1):
    def make_loader(idx, cfg):
        return DataLoader(TensorDataset(splits["x"][idx], splits["y"][idx]),
                          batch_size=int(cfg["batch_size"]), shuffle=True)

    return PETRecipe(
        make_model=lambda cfg: nn.Linear(dim, classes),
        make_optimizer=lambda params, cfg: optim.SGD(params, lr=cfg["learning_rate"]),
        make_loader=make_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=epochs,
    )


PRIVATE = {"noise_multiplier": 1.0, "max_grad_norm": 1.0, "learning_rate": 0.1, "batch_size": 64}
NONPRIVATE = {"noise_multiplier": 0.0, "max_grad_norm": 1.0, "learning_rate": 0.1, "batch_size": 64}


class TestTrainWithDPSGD:
    def test_private_path_records_finite_epsilon(self):
        splits = _toy_splits()
        model = train_with_dpsgd(_toy_recipe(splits), PRIVATE, splits["target_train"], "cpu")
        assert np.isfinite(model.campaign_extras["epsilon"])
        assert model.campaign_extras["epsilon"] > 0

    def test_nonprivate_path_is_epsilon_inf(self):
        splits = _toy_splits()
        model = train_with_dpsgd(_toy_recipe(splits), NONPRIVATE, splits["target_train"], "cpu")
        assert model.campaign_extras["epsilon"] == float("inf")

    def test_physical_batch_cap_matches_uncapped_accounting(self):
        splits = _toy_splits()
        recipe = _toy_recipe(splits)
        m_small = train_with_dpsgd(recipe, PRIVATE, splits["target_train"], "cpu", max_physical_batch=16)
        m_large = train_with_dpsgd(recipe, PRIVATE, splits["target_train"], "cpu", max_physical_batch=512)
        # The cap is a memory measure only: the accounting must be identical.
        assert m_small.campaign_extras["epsilon"] == pytest.approx(m_large.campaign_extras["epsilon"], rel=1e-6)

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


class TestConfidenceSignal:
    def test_multiclass_prefers_true_class(self):
        model = nn.Identity()  # "logits" are the inputs themselves
        x = torch.eye(3).repeat(4, 1) * 5  # strongly peaked one-hot logits
        y = torch.arange(3).repeat(4)
        phi = confidence_signal(model, x, y, "cpu", output_kind="logits")
        y_wrong = (y + 1) % 3
        phi_wrong = confidence_signal(model, x, y_wrong, "cpu", output_kind="logits")
        assert phi.min() > phi_wrong.max()

    def test_binary_probs_symmetry(self):
        model = nn.Identity()
        p = torch.tensor([[0.9], [0.9]])
        y = torch.tensor([[1.0], [0.0]])
        phi = confidence_signal(model, p, y, "cpu", output_kind="binary_probs")
        assert phi[0] == pytest.approx(-phi[1])  # log(0.9/0.1) vs log(0.1/0.9)

    def test_binary_logits_signed_by_label(self):
        model = nn.Identity()
        logit = torch.tensor([[2.0], [2.0]])
        y = torch.tensor([[1.0], [0.0]])
        phi = confidence_signal(model, logit, y, "cpu", output_kind="binary_logits")
        assert phi[0] == pytest.approx(2.0)   # confident and correct
        assert phi[1] == pytest.approx(-2.0)  # confident and wrong


class TestBuildCampaignFns:
    def test_end_to_end_on_toy_data(self):
        splits = _toy_splits()
        train_fn, utility_fn, attack_fn = build_campaign_fns(
            _toy_recipe(splits), splits, n_refs=1, device="cpu")
        model = train_fn(NONPRIVATE)
        acc = utility_fn(model)
        assert 0.0 <= acc <= 1.0
        scores = attack_fn(model, NONPRIVATE)
        assert isinstance(scores, AttackScores)
        assert len(scores.member_scores) == 100
        assert len(scores.nonmember_scores) == 100

    def test_missing_split_keys_rejected(self):
        splits = _toy_splits()
        del splits["ref_pool"]
        with pytest.raises(KeyError):
            build_campaign_fns(_toy_recipe(splits), splits, n_refs=1, device="cpu")

    def test_unknown_utility_metric_rejected(self):
        splits = _toy_splits()
        _, utility_fn, _ = build_campaign_fns(
            _toy_recipe(splits), splits, n_refs=1, device="cpu", utility_metric="f1")
        model = train_with_dpsgd(_toy_recipe(splits), NONPRIVATE, splits["target_train"], "cpu")
        with pytest.raises(ValueError):
            utility_fn(model)

    def test_callable_utility_metric(self):
        splits = _toy_splits()
        _, utility_fn, _ = build_campaign_fns(
            _toy_recipe(splits), splits, n_refs=1, device="cpu",
            utility_metric=lambda model, x, y, device: 0.42)
        model = train_with_dpsgd(_toy_recipe(splits), NONPRIVATE, splits["target_train"], "cpu")
        assert utility_fn(model) == 0.42


def _batchnorm_recipe(splits, dim=8, classes=3, epochs=1):
    """A model Opacus rejects as submitted: BatchNorm mixes samples within a batch."""
    def make_loader(idx, cfg):
        return DataLoader(TensorDataset(splits["x"][idx], splits["y"][idx]),
                          batch_size=int(cfg["batch_size"]), shuffle=True)

    return PETRecipe(
        make_model=lambda cfg: nn.Sequential(
            nn.Linear(dim, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.Linear(16, classes)),
        make_optimizer=lambda params, cfg: optim.SGD(params, lr=cfg["learning_rate"]),
        make_loader=make_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=epochs,
    )


class TestOpacusCompatibility:
    def test_batchnorm_model_trains_under_dpsgd(self):
        # Before the ModuleValidator pass this raised straight out of make_private.
        splits = _toy_splits()
        model = train_with_dpsgd(_batchnorm_recipe(splits), PRIVATE, splits["target_train"], "cpu")
        assert np.isfinite(model.campaign_extras["epsilon"])

    def test_batchnorm_is_replaced_and_inplace_disabled(self):
        fixed = make_opacus_compatible(
            nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True)))
        assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in fixed.modules())
        assert not any(getattr(m, "inplace", False) for m in fixed.modules())

    def test_nonprivate_path_leaves_batchnorm_alone(self):
        # eps = inf means no Opacus, so there is no reason to rewrite the user's model.
        splits = _toy_splits()
        model = train_with_dpsgd(_batchnorm_recipe(splits), NONPRIVATE, splits["target_train"], "cpu")
        assert any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())

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
        splits = {"x": x, "y": y, "target_train": np.arange(64), "ref_pool": np.arange(64),
                  "audit_members": np.arange(8), "audit_nonmembers": np.arange(8), "utility_eval": np.arange(8)}

        recipe = PETRecipe(
            make_model=lambda cfg: nn.Sequential(
                torchvision.models.resnet.BasicBlock(4, 4),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)),
            make_optimizer=lambda params, cfg: optim.SGD(params, lr=cfg["learning_rate"]),
            make_loader=lambda idx, cfg: DataLoader(
                TensorDataset(splits["x"][idx], splits["y"][idx]), batch_size=int(cfg["batch_size"]), shuffle=True),
            criterion=nn.CrossEntropyLoss(),
            epochs=1,
        )
        model = train_with_dpsgd(recipe, PRIVATE, splits["target_train"], "cpu")
        assert np.isfinite(model.campaign_extras["epsilon"])


class TestAccountant:
    def test_accountant_is_configurable_and_changes_epsilon(self):
        splits = _toy_splits()
        recipe = _toy_recipe(splits)
        prv = train_with_dpsgd(recipe, PRIVATE, splits["target_train"], "cpu", accountant="prv")
        rdp = train_with_dpsgd(recipe, PRIVATE, splits["target_train"], "cpu", accountant="rdp")
        # PRV is the tighter bound, so it must not report a larger epsilon than RDP.
        assert prv.campaign_extras["epsilon"] < rdp.campaign_extras["epsilon"]

    def test_default_accountant_is_prv(self):
        splits = _toy_splits()
        recipe = _toy_recipe(splits)
        default = train_with_dpsgd(recipe, PRIVATE, splits["target_train"], "cpu")
        prv = train_with_dpsgd(recipe, PRIVATE, splits["target_train"], "cpu", accountant="prv")
        assert default.campaign_extras["epsilon"] == pytest.approx(prv.campaign_extras["epsilon"], rel=1e-9)

    def test_build_campaign_fns_threads_accountant_through(self):
        splits = _toy_splits()
        train_fn, _, _ = build_campaign_fns(
            _toy_recipe(splits), splits, n_refs=1, device="cpu", accountant="rdp")
        direct = train_with_dpsgd(_toy_recipe(splits), PRIVATE, splits["target_train"], "cpu", accountant="rdp")
        assert train_fn(PRIVATE).campaign_extras["epsilon"] == pytest.approx(
            direct.campaign_extras["epsilon"], rel=1e-9)
