#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""End-to-end test of the DP-SGD search objective on a tiny tabular problem (CPU, seconds)."""

import pickle

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

from leakpro.optimization.adapters import recipe_from_module
from leakpro.optimization.objective import (
    AuditTarget,
    DPSGDAuditHandler,
    config_dir,
    majority_rate,
    make_dpsgd_objective,
    materialize_population,
    materialize_split,
    reaudit_trial,
    resolution_warning,
    tpr_uncertainty,
    trial_dir_for,
)
from leakpro.reporting.mia_result import MIAResult
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
    target = AuditTarget(module_path=str(arch), model_class="Net", dataset_name="blobs", data_modality="tabular",
                         init_params={"num_classes": 3})
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
        # The shadow models are built with the target's constructor kwargs even
        # when the architecture does not store them as attributes.
        audit_cfg = yaml.safe_load((trial_dir / "audit.yaml").read_text())
        assert audit_cfg["shadow_model"] == {"init_params": {"num_classes": 3}}

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


class TestTrialDirectories:
    def test_identical_configs_get_separate_directories_inside_a_study(self, run):  # noqa: ANN001, ANN201
        """Two trials with the same knobs are two models; the second must not overwrite the first."""
        config = {"noise_multiplier": 0.0, "max_grad_norm": 1.0, "learning_rate": 0.05, "batch_size": 32}
        first = run["objective"](config, trial_number=3)
        second = run["objective"](config, trial_number=4)
        assert first.extras["trial_dir"] == str(trial_dir_for(run["out"], 3))
        assert second.extras["trial_dir"] == str(trial_dir_for(run["out"], 4))
        assert (trial_dir_for(run["out"], 3) / "target" / "target_model.pkl").exists()
        assert (trial_dir_for(run["out"], 4) / "target" / "target_model.pkl").exists()

    def test_objective_advertises_the_trial_number_keyword(self, run):  # noqa: ANN001, ANN201
        import inspect
        assert "trial_number" in inspect.signature(run["objective"]).parameters


class TestUtilityGate:
    def test_gate_is_the_majority_class_rate(self):  # noqa: ANN201
        assert majority_rate(np.array([0, 0, 0, 0, 1])) == pytest.approx(0.8)
        assert majority_rate(torch.tensor([0, 1, 2])) == pytest.approx(1 / 3)

    def test_constant_predictor_is_gated_on_imbalanced_labels(self, tmp_path):  # noqa: ANN001, ANN201
        """A model that always predicts the majority class scores 0.8 on 80/20 labels; it must be gated."""
        from leakpro.optimization.objective import train_and_audit
        from leakpro.optimization.training import PETRecipe

        seed_everything(0)
        rng = np.random.default_rng(0)
        n = 400
        y = (rng.random(n) < 0.2).astype(np.int64)
        x = rng.normal(size=(n, 4)).astype(np.float32)   # no signal: the model can only learn the prior
        dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
        population = materialize_population(dataset, tmp_path, n_target=150, n_test=100, seed=0)
        arch = tmp_path / "arch.py"
        arch.write_text(ARCH)
        with open(population.path, "rb") as f:
            pop = pickle.load(f)
        recipe = recipe_from_module(arch, pop, epochs=3, init_params={"num_classes": 2})
        target = AuditTarget(module_path=str(arch), model_class="Net", data_modality="tabular",
                             init_params={"num_classes": 2})
        result, mia = train_and_audit(recipe, population, target, tmp_path / "t", {
            "noise_multiplier": 0.0, "max_grad_norm": 1.0, "learning_rate": 0.05, "batch_size": 32},
            proxy_fpr=PROXY_FPR, rmia_config={"num_shadow_models": 1}, seed=0, device="cpu")
        assert result.extras["gate"] == pytest.approx(majority_rate(pop.targets[population.test_indices]))
        assert result.extras["gate"] > 0.5, "the old 1/num_classes gate would have let this through"
        assert result.tpr is None and result.extras["gated"] is True and mia is None


class TestUncertainty:
    def test_counts_and_interval_come_from_the_result(self):  # noqa: ANN201
        labels = np.array([1] * 200 + [0] * 300)
        scores = np.concatenate([np.linspace(1, 2, 200), np.linspace(0, 1, 300)])
        result = MIAResult.from_full_scores(true_membership=labels, signal_values=scores, result_name="rmia")
        u = tpr_uncertainty(result, tpr=0.25, realized_fpr=0.01)
        assert u["n_members"] == 200 and u["n_non_members"] == 300
        assert u["n_nonmember_false_positives"] == 3
        lo, hi = u["attack_tpr_ci"]
        assert lo < 0.25 < hi

    def test_objective_records_the_interval(self, run):  # noqa: ANN001, ANN201
        config = {"noise_multiplier": 0.0, "max_grad_norm": 1.0, "learning_rate": 0.05, "batch_size": 32}
        result = run["objective"](config, trial_number=0)
        assert result.tpr is not None
        assert result.extras["n_members"] == N_TARGET
        lo, hi = result.extras["attack_tpr_ci"]
        assert 0.0 <= lo <= result.tpr <= hi <= 1.0
        assert result.extras["n_shadow"] == 2


class TestMaterializeSplit:
    def test_members_first_then_non_members_and_utility_positions(self, tmp_path):  # noqa: ANN001, ANN201
        dataset = _separable_dataset(n=300)
        members = list(range(10, 110))
        non_members = list(range(200, 300))
        population = materialize_split(dataset, tmp_path, members=members, non_members=non_members,
                                       utility=non_members[:40])
        with open(population.path, "rb") as f:
            pop = pickle.load(f)
        assert len(pop) == 200
        assert population.train_indices == list(range(100))
        assert population.test_indices == list(range(100, 200))
        assert population.utility_split == list(range(100, 140))
        # Row 0 of the file is dataset[10]; row 100 is dataset[200].
        assert torch.equal(pop.data[0], dataset[10][0]) and torch.equal(pop.data[100], dataset[200][0])

    def test_overlap_and_foreign_utility_are_rejected(self, tmp_path):  # noqa: ANN001, ANN201
        dataset = _separable_dataset(n=100)
        with pytest.raises(ValueError, match="overlap"):
            materialize_split(dataset, tmp_path, members=[0, 1], non_members=[1, 2])
        with pytest.raises(ValueError, match="subset of non_members"):
            materialize_split(dataset, tmp_path, members=[0, 1], non_members=[2, 3], utility=[0])

    def test_unequal_pools_are_written_with_a_warning(self, tmp_path, caplog):  # noqa: ANN001, ANN201
        dataset = _separable_dataset(n=100)
        population = materialize_split(dataset, tmp_path, members=list(range(50)), non_members=list(range(50, 70)))
        assert len(population.test_indices) == 20
        assert "not 2 x members" in caplog.text
