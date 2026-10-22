#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""End-to-end test of the DP-SGD search objective on a tiny tabular problem (CPU, seconds)."""

import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from leakpro.optimization.adapters import recipe_from_module
from leakpro.optimization.objective import (
    AuditTarget,
    DPSGDAuditHandler,
    config_dir,
    make_dpsgd_objective,
    materialize_population,
    reaudit_trial,
    resolution_warning,
)
from leakpro.optimization.search import ObjectiveResult
from leakpro.utils.seed import seed_everything

ARCH = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, num_classes))
    def forward(self, x):
        return self.fc(x)
"""

N_TARGET = 120
N_TEST = 100
PROXY_FPR = 0.1


def _separable_dataset(n: int = 400, classes: int = 3, seed: int = 0) -> TensorDataset:
    """Gaussian blobs: learnable in a few epochs, so the utility gate passes."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, size=n)
    # Overlapping blobs: learnable, but not so separable that every softmax
    # saturates to 1.0 (then all RMIA scores tie and the audit is degenerate).
    means = np.eye(classes, 4)[y] * 2.0
    x = means + rng.normal(scale=1.5, size=(n, 4))
    return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y))


@pytest.fixture()
def run(tmp_path):  # noqa: ANN001, ANN201
    # optimize() re-seeds every trial from (seed, trial number); do the same here
    # so a test's model does not depend on which tests ran before it.
    seed_everything(0)
    arch = tmp_path / "arch.py"
    arch.write_text(ARCH)
    dataset = _separable_dataset()
    population = materialize_population(dataset, tmp_path, n_target=N_TARGET, n_test=N_TEST, seed=1)
    with open(population.path, "rb") as f:
        pop = pickle.load(f)
    recipe = recipe_from_module(arch, pop, epochs=5, init_params={"num_classes": 3})
    target = AuditTarget(module_path=str(arch), model_class="Net", dataset_name="blobs", data_modality="tabular")
    objective = make_dpsgd_objective(
        recipe, population, target, tmp_path,
        proxy_fpr=PROXY_FPR, rmia_config={"num_shadow_models": 2}, seed=0, device="cpu",
    )
    return {"objective": objective, "population": population, "target": target, "out": tmp_path}


class TestPopulation:
    def test_is_twice_the_target_and_roles_are_disjoint(self, tmp_path):  # noqa: ANN001, ANN201
        population = materialize_population(_separable_dataset(), tmp_path, n_target=N_TARGET, n_test=N_TEST)
        with open(population.path, "rb") as f:
            pop = pickle.load(f)
        assert isinstance(pop, DPSGDAuditHandler.UserDataset)
        assert len(pop) == 2 * N_TARGET
        assert not set(population.train_indices) & set(population.test_indices)
        assert population.num_classes == 3

    def test_reads_through_getitem(self, tmp_path):  # noqa: ANN001, ANN201
        class Scaled(TensorDataset):
            def __getitem__(self, idx):  # noqa: ANN001, ANN204
                x, y = super().__getitem__(idx)
                return x * 0.0 + 7.0, y

        base = _separable_dataset()
        population = materialize_population(Scaled(*base.tensors), tmp_path, n_target=50)
        with open(population.path, "rb") as f:
            pop = pickle.load(f)
        assert torch.all(pop.data == 7.0)

    def test_rejects_a_dataset_too_small_for_the_mimicry_rule(self, tmp_path):  # noqa: ANN001, ANN201
        with pytest.raises(ValueError, match="at least"):
            materialize_population(_separable_dataset(n=100), tmp_path, n_target=60)


class TestResolutionWarning:
    def test_untabulated_fpr_is_rejected(self):  # noqa: ANN201
        assert "not a tabulated" in resolution_warning(10_000, 0.05)

    def test_too_few_events_is_flagged(self):  # noqa: ANN201
        assert "unresolvable" in resolution_warning(200, 0.01)

    def test_enough_events_is_fine(self):  # noqa: ANN201
        assert resolution_warning(2000, 0.01) is None


class TestObjective:
    def test_trains_audits_and_resolves_tpr(self, run):  # noqa: ANN001, ANN201
        config = {"noise_multiplier": 0.0, "max_grad_norm": 1.0, "learning_rate": 0.05, "batch_size": 32}
        result = run["objective"](config)

        assert isinstance(result, ObjectiveResult)
        assert result.utility > 1.0 / 3  # learned something: the gate passed
        assert result.tpr is not None and 0.0 <= result.tpr <= 1.0
        assert result.extras["epsilon"] == float("inf")  # non-private anchor
        assert result.extras["tuning_accounted"] is False
        assert result.extras["realized_fpr"] is not None

        trial_dir = config_dir(run["out"], config)
        assert (trial_dir / "target" / "target_model.pkl").exists()
        assert (trial_dir / "target" / "model_metadata.pkl").exists()
        assert (trial_dir / "target" / "dpsgd_dic.pkl").exists()
        assert (trial_dir / "audit.yaml").exists()
        assert result.extras["trial_dir"] == str(trial_dir)

    def test_private_run_reports_a_finite_epsilon(self, run):  # noqa: ANN001, ANN201
        config = {"noise_multiplier": 1.0, "max_grad_norm": 1.0, "learning_rate": 0.05, "batch_size": 32}
        result = run["objective"](config)
        assert np.isfinite(result.extras["epsilon"])
        assert result.extras["accountant"] == "prv"

    def test_reaudit_reads_the_saved_target_back(self, run):  # noqa: ANN001, ANN201
        config = {"noise_multiplier": 0.0, "max_grad_norm": 1.0, "learning_rate": 0.05, "batch_size": 32}
        first = run["objective"](config)
        assert first.tpr is not None
        trial_dir = config_dir(run["out"], config)
        result = reaudit_trial(trial_dir, population_path=run["population"].path, target=run["target"],
                               rmia_config={"num_shadow_models": 3}, seed=0)
        assert result.result_name.upper() == "RMIA"
        assert (trial_dir / "verification_audit.yaml").exists()
        assert (trial_dir / "verification").exists()

    def test_reaudit_of_a_pruned_trial_is_refused(self, run, tmp_path):  # noqa: ANN001, ANN201
        with pytest.raises(FileNotFoundError, match="no saved target"):
            reaudit_trial(tmp_path / "nowhere", population_path=run["population"].path,
                          target=run["target"], rmia_config={})
