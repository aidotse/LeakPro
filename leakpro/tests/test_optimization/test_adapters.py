#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for building a recipe and splits from an architecture file (CPU, tiny models)."""

import numpy as np
import pytest
import torch

from leakpro.optimization import Campaign, Knob, KnobSpace, build_campaign_fns
from leakpro.optimization.adapters import (
    carve_splits,
    detect_binary,
    find_model_class,
    recipe_from_module,
)
from leakpro.input_handler.user_imports import import_module_from_file
from leakpro.optimization.training import REQUIRED_SPLIT_KEYS

MULTICLASS = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(6, num_classes)
    def forward(self, x):
        return self.fc(x)
"""

BINARY = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 1)
    def forward(self, x):
        return self.fc(x)
"""

TWO_MODELS = MULTICLASS + """
class Other(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 2)
    def forward(self, x):
        return self.fc(x)
"""


@pytest.fixture()
def arch(tmp_path):
    def write(source: str):
        path = tmp_path / "arch.py"
        path.write_text(source)
        return path
    return write


def _xy(n=200, dim=6, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n, dim)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, classes, size=n))
    return x, y


class TestModelDiscovery:
    def test_single_class_is_found_without_naming_it(self, arch):
        assert find_model_class(import_module_from_file(str(arch(MULTICLASS)))).__name__ == "Net"

    def test_ambiguous_module_requires_a_name(self, arch):
        module = import_module_from_file(str(arch(TWO_MODELS)))
        with pytest.raises(ValueError, match="Cannot choose a model class"):
            find_model_class(module)
        assert find_model_class(module, "Other").__name__ == "Other"

    def test_unknown_class_name_rejected(self, arch):
        # user_imports.get_class_from_module raises ValueError, not KeyError.
        with pytest.raises(ValueError, match="not found in module"):
            find_model_class(import_module_from_file(str(arch(MULTICLASS))), "Missing")

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_module_from_file(str(tmp_path / "nope.py"))

    def test_repeated_loads_reuse_the_cached_module(self, arch):
        # user_imports.import_module_from_file caches by module name and checks
        # the path, so the same arch loaded twice is one import, not two.
        path = arch(MULTICLASS)
        assert import_module_from_file(str(path)) is import_module_from_file(str(path))


class TestDetectBinary:
    def test_single_logit_head(self, arch):
        x, _ = _xy()
        assert detect_binary(arch(BINARY), x) is True

    def test_multiclass_head(self, arch):
        x, _ = _xy()
        assert detect_binary(arch(MULTICLASS), x) is False


class TestRecipeFromModule:
    def test_multiclass_recipe_shape(self, arch):
        x, y = _xy()
        recipe = recipe_from_module(arch(MULTICLASS), x, y.long(), epochs=1)
        assert recipe.output_kind == "logits"
        assert isinstance(recipe.criterion, torch.nn.CrossEntropyLoss)

    def test_binary_recipe_shape(self, arch):
        x, y = _xy(classes=2)
        recipe = recipe_from_module(arch(BINARY), x, y.float(), epochs=1, binary=True)
        assert recipe.output_kind == "binary_logits"
        assert isinstance(recipe.criterion, torch.nn.BCEWithLogitsLoss)

    def test_each_call_returns_an_untrained_model(self, arch):
        # A campaign that resumed a trained model would silently measure the
        # wrong thing, so make_model must never hand back the same object.
        x, y = _xy()
        recipe = recipe_from_module(arch(MULTICLASS), x, y.long(), epochs=1)
        assert recipe.make_model({}) is not recipe.make_model({})

    def test_loader_honours_sampled_batch_size(self, arch):
        x, y = _xy()
        recipe = recipe_from_module(arch(MULTICLASS), x, y.long(), epochs=1)
        loader = recipe.make_loader(np.arange(64), {"batch_size": 16})
        assert next(iter(loader))[0].shape[0] == 16

    def test_optimizer_honours_sampled_learning_rate(self, arch):
        x, y = _xy()
        recipe = recipe_from_module(arch(MULTICLASS), x, y.long(), epochs=1, optimizer_name="sgd")
        opt = recipe.make_optimizer(recipe.make_model({}).parameters(), {"learning_rate": 0.07})
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.07)

    def test_unknown_optimizer_rejected(self, arch):
        x, y = _xy()
        # Every torch.optim optimizer is valid now that the lookup comes from
        # user_imports.get_optimizer_mapping, so "rmsprop" is accepted; only a
        # name torch does not have is rejected.
        assert recipe_from_module(arch(MULTICLASS), x, y.long(), epochs=1, optimizer_name="rmsprop")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            recipe_from_module(arch(MULTICLASS), x, y.long(), epochs=1, optimizer_name="not_an_optimizer")


class TestCarveSplits:
    def test_produces_every_required_key(self):
        x, y = _xy(n=400)
        splits = carve_splits(x, y)
        assert all(k in splits for k in REQUIRED_SPLIT_KEYS)

    def test_roles_are_disjoint_where_they_must_be(self):
        x, y = _xy(n=400)
        s = carve_splits(x, y)
        # The reference pool must never touch the target's training data,
        # otherwise the "reference" models are partly trained on members.
        assert not set(s["target_train"]) & set(s["ref_pool"])
        # Nonmembers must be non-members, and must not double as the utility split.
        assert not set(s["audit_nonmembers"]) & set(s["target_train"])
        assert not set(s["audit_nonmembers"]) & set(s["utility_eval"])
        # Members are audited *because* they were trained on.
        assert set(s["audit_members"]) <= set(s["target_train"])

    def test_is_deterministic_for_a_seed(self):
        x, y = _xy(n=400)
        assert np.array_equal(carve_splits(x, y, seed=3)["target_train"],
                              carve_splits(x, y, seed=3)["target_train"])
        assert not np.array_equal(carve_splits(x, y, seed=3)["target_train"],
                                  carve_splits(x, y, seed=4)["target_train"])

    def test_audit_size_capped_by_available_data(self):
        x, y = _xy(n=100)
        s = carve_splits(x, y, audit_size=10_000)
        assert len(s["audit_members"]) == len(s["audit_nonmembers"])
        assert len(s["audit_members"]) <= len(s["target_train"])

    def test_oversubscribed_fractions_rejected(self):
        x, y = _xy(n=400)
        with pytest.raises(ValueError, match="less than 1"):
            carve_splits(x, y, target_fraction=0.6, ref_fraction=0.5)

    @pytest.mark.parametrize("n", [2, 3])
    def test_empty_role_rejected(self, n):
        # carve_splits guards structure only: a role that comes out empty is an
        # error. "Non-empty but far too small to resolve an FPR" is a separate
        # judgement, and belongs to validation.resolution_warning.
        x, y = _xy(n=n)
        with pytest.raises(ValueError, match="too small"):
            carve_splits(x, y, target_fraction=0.45, ref_fraction=0.45)

    def test_smallest_workable_dataset_is_accepted(self):
        x, y = _xy(n=400)
        s = carve_splits(x, y, target_fraction=0.45, ref_fraction=0.45)
        assert all(len(s[k]) > 0 for k in ("target_train", "ref_pool", "audit_members",
                                           "audit_nonmembers", "utility_eval"))


class TestAdapterDrivesACampaign:
    def test_end_to_end_from_an_architecture_file(self, arch, tmp_path):
        # The whole point of the adapter: a .py file plus a tensor dataset is
        # enough to run a campaign, with no hand-written callables.
        x, y = _xy(n=400)
        recipe = recipe_from_module(arch(MULTICLASS), x, y.long(), epochs=1)
        splits = carve_splits(x, y.long())
        fns = build_campaign_fns(recipe, splits, n_refs=1, device="cpu")
        space = KnobSpace([
            Knob("noise_multiplier", 0.5, 2.0, log_scale=True),
            Knob("max_grad_norm", 0.5, 2.0, log_scale=True),
            Knob("learning_rate", 1e-3, 1e-1, log_scale=True),
            Knob("batch_size", 32, 64, integer=True),
        ])
        records = Campaign(*fns, knob_space=space, output_dir=tmp_path / "out").run(2)

        assert len(records) == 2
        assert all(np.isfinite(r["epsilon"]) for r in records)
        assert all(0.0 <= r["attack_tpr"] <= 1.0 for r in records)
        assert (tmp_path / "out" / "evaluations.jsonl").exists()
