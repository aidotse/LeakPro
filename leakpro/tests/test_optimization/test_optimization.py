#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for leakpro.optimization: the BO search, the RMIA bridge, frontier, validation.

Model training is never exercised here — the objective is a cheap synthetic
function and the RMIA bridge is tested against a real MIAResult built from
synthetic scores. The end-to-end pipeline (train target -> real RMIA audit) is
exercised by the CIFAR example, not by unit tests.
"""

import numpy as np
import optuna
import pytest

from leakpro.optimization import (
    Knob,
    KnobSpace,
    ObjectiveResult,
    clopper_pearson_ci,
    default_dpsgd_space,
    optimize,
    pareto_trials,
    plot_frontier,
    proxy_agreement,
    tpr_at_fixed_fpr,
    validate_frontier,
)
from leakpro.reporting.mia_result import MIAResult

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Knob space
# --------------------------------------------------------------------------- #
class TestKnobs:
    def test_suggest_respects_bounds_and_fixed(self):
        space = default_dpsgd_space().fix("batch_size", 128)
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))

        def objective(trial):
            config = space.suggest(trial)
            assert config["batch_size"] == 128  # fixed knob emitted verbatim
            assert 0.4 <= config["noise_multiplier"] <= 8.0
            assert 1e-5 <= config["learning_rate"] <= 1e-1
            return 0.0

        study.optimize(objective, n_trials=8)
        # batch_size is fixed, so it must not appear as a searched parameter.
        assert "batch_size" not in study.trials[0].params
        assert "noise_multiplier" in study.trials[0].params

    def test_log_scale_covers_decades(self):
        space = KnobSpace([Knob("lr", 1e-5, 1e-1, log_scale=True)])
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(lambda t: space.suggest(t)["lr"], n_trials=256)
        values = [t.params["lr"] for t in study.trials]
        assert min(values) < 1e-3 < max(values)

    def test_narrow_and_fix_and_invalid(self):
        space = default_dpsgd_space().narrow("noise_multiplier", 1.0, 2.0)
        knob = next(k for k in space.knobs if k.name == "noise_multiplier")
        assert (knob.low, knob.high) == (1.0, 2.0)
        with pytest.raises(KeyError):
            space.fix("does_not_exist", 1.0)
        with pytest.raises(ValueError):
            Knob("bad", 5.0, 1.0)
        with pytest.raises(ValueError):
            Knob("bad_log", -1.0, 1.0, log_scale=True)


# --------------------------------------------------------------------------- #
# The search actually optimizes: proposals depend on observed results.
# --------------------------------------------------------------------------- #
class TestSearchIsModelBased:
    """The defining property the old Sobol sweep lacked."""

    @staticmethod
    def _run(tmp_path, name, sign):
        space = KnobSpace([Knob("a", -5.0, 5.0), Knob("b", -5.0, 5.0)])

        def objective_fn(config):
            # utility and tpr are (signed) functions of the knobs, so the two
            # "worlds" (sign = +1 / -1) return opposite results for the same config.
            return ObjectiveResult(utility=sign * config["a"], tpr=sign * config["b"])

        study = optimize(objective_fn, space, tmp_path / name, n_trials=25, seed=0, study_name=name)
        return [(round(t.params["a"], 3), round(t.params["b"], 3)) for t in study.trials]

    def test_proposals_diverge_when_results_differ(self, tmp_path):
        world_a = self._run(tmp_path, "world_a", sign=1.0)
        world_b = self._run(tmp_path, "world_b", sign=-1.0)
        # Startup trials are sampled before any model is fit and may coincide, but
        # once results feed back the two searches must propose different points.
        assert world_a[:5] == world_b[:5]  # random startup, seed-shared
        assert world_a[10:] != world_b[10:]  # results steer the search apart

    def test_resume_skips_finished_trials(self, tmp_path):
        space = KnobSpace([Knob("a", -5.0, 5.0)])
        calls = {"n": 0}

        def objective_fn(config):
            calls["n"] += 1
            return ObjectiveResult(utility=config["a"], tpr=-config["a"])

        optimize(objective_fn, space, tmp_path / "c", n_trials=4, seed=0, study_name="c")
        assert calls["n"] == 4
        # Same output dir + study name: the first 4 trials are already in the db.
        study = optimize(objective_fn, space, tmp_path / "c", n_trials=6, seed=0, study_name="c")
        assert calls["n"] == 6  # only 2 more objective evaluations
        assert len([t for t in study.trials if t.state.is_finished()]) == 6

    def test_gated_trials_are_pruned_off_the_frontier(self, tmp_path):
        """tpr=None means the utility gate failed: pruned, never on the Pareto front."""
        space = KnobSpace([Knob("a", 0.0, 1.0)])

        def objective_fn(config):
            if config["a"] < 0.5:  # "didn't learn" region
                return ObjectiveResult(utility=0.1, tpr=None, extras={"gated": True})
            return ObjectiveResult(utility=config["a"], tpr=config["a"])

        study = optimize(objective_fn, space, tmp_path / "g", n_trials=12, seed=0, study_name="g")
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        assert pruned, "some trials should hit the gate"
        front_numbers = {t.number for t in pareto_trials(study)}
        assert not front_numbers & {t.number for t in pruned}
        # Pruned trials still consume budget (each one trained a model).
        assert len([t for t in study.trials if t.state.is_finished()]) == 12

    def test_anchor_is_evaluated_first(self, tmp_path):
        space = default_dpsgd_space()
        anchor = {"noise_multiplier": 0.4, "max_grad_norm": 10.0, "learning_rate": 0.05, "batch_size": 128}
        seen = []

        def objective_fn(config):
            seen.append(config)
            return ObjectiveResult(utility=0.5, tpr=0.5)

        optimize(objective_fn, space, tmp_path / "a", n_trials=3, seed=0,
                 anchors=[anchor], study_name="a")
        assert seen[0]["noise_multiplier"] == pytest.approx(0.4)
        assert seen[0]["batch_size"] == 128


# --------------------------------------------------------------------------- #
# The RMIA bridge: TPR is read from the real MIAResult, not recomputed.
# --------------------------------------------------------------------------- #
def _mia_result(member_mu: float, n: int = 500) -> MIAResult:
    rng = np.random.default_rng(0)
    members = rng.normal(member_mu, 1.0, n)
    nonmembers = rng.normal(0.0, 1.0, n)
    true = np.concatenate([np.ones(n), np.zeros(n)])
    signal = np.concatenate([members, nonmembers])
    return MIAResult.from_full_scores(true_membership=true, signal_values=signal,
                                      result_name="RMIA", metadata={})


class TestRmiaBridge:
    def test_tpr_matches_fixed_fpr_table(self):
        result = _mia_result(member_mu=1.5)
        # The bridge must return exactly what MIAResult already reports.
        assert tpr_at_fixed_fpr(result, 0.01) == pytest.approx(result.fixed_fpr_table["TPR@1%FPR"])
        assert tpr_at_fixed_fpr(result, 0.001) == pytest.approx(result.fixed_fpr_table["TPR@0.1%FPR"])

    def test_more_leakage_means_higher_tpr(self):
        low = tpr_at_fixed_fpr(_mia_result(member_mu=0.2), 0.01)
        high = tpr_at_fixed_fpr(_mia_result(member_mu=3.0), 0.01)
        assert high > low

    def test_unavailable_fpr_raises(self):
        with pytest.raises(ValueError, match="No TPR at FPR"):
            tpr_at_fixed_fpr(_mia_result(1.0), 0.5)  # 50% FPR is not in the table

    def test_clopper_pearson_edges(self):
        assert clopper_pearson_ci(0, 100)[0] == 0.0
        assert clopper_pearson_ci(100, 100)[1] == 1.0
        low, high = clopper_pearson_ci(5, 100)
        assert 0.0 < low < 0.05 < high < 1.0


# --------------------------------------------------------------------------- #
# Frontier + validation
# --------------------------------------------------------------------------- #
def _study_with_front(tmp_path):
    space = KnobSpace([Knob("a", 0.0, 1.0)])

    def objective_fn(config):
        # utility increases with a, tpr also increases with a -> a genuine
        # trade-off, so several points are non-dominated.
        return ObjectiveResult(utility=config["a"], tpr=config["a"])

    return optimize(objective_fn, space, tmp_path / "s", n_trials=12, seed=0, study_name="s")


class TestFrontier:
    def test_pareto_trials_are_non_dominated(self, tmp_path):
        study = _study_with_front(tmp_path)
        front = pareto_trials(study)
        assert front
        for a in front:
            for b in front:
                if a is b:
                    continue
                # No trial dominates another on the front.
                assert not (b.values[0] >= a.values[0] and b.values[1] <= a.values[1]
                            and b.values != a.values)

    def test_plot_writes_file(self, tmp_path):
        study = _study_with_front(tmp_path)
        out = plot_frontier(study, tmp_path / "frontier.png")
        assert out.exists() and out.stat().st_size > 0


class TestValidation:
    def test_validate_frontier_reports_each_fpr(self, tmp_path):
        study = _study_with_front(tmp_path)

        def revalidate_fn(params):
            # Stronger audit stand-in: leakage grows with the knob.
            return _mia_result(member_mu=0.5 + 3 * params["a"])

        rows = validate_frontier(study, revalidate_fn, report_fprs=(0.001, 0.01))
        assert rows
        for row in rows:
            assert "tpr_at_0.01" in row["revalidated"]
            assert "tpr_at_0.001" in row["revalidated"]
            assert "selection_bias" in row

    def test_revalidation_receives_the_full_config_with_fixed_knobs(self, tmp_path):
        """Regression: trial.params lacks fixed knobs, so anything keyed on the
        full configuration (per-trial dirs, re-audits) broke as soon as a knob
        was pinned. Validation must hand revalidate_fn the resolved config."""
        space = KnobSpace([Knob("a", 0.0, 1.0), Knob("batch_size", 32, 512)]).fix("batch_size", 128)

        def objective_fn(config):
            assert config["batch_size"] == 128
            return ObjectiveResult(utility=config["a"], tpr=config["a"])

        study = optimize(objective_fn, space, tmp_path / "f", n_trials=6, seed=0, study_name="f")
        # The search records the resolved config on every trial...
        for trial in study.trials:
            assert trial.user_attrs["config"]["batch_size"] == 128
            assert "batch_size" not in trial.params  # what made trial.params insufficient

        # ...and validation passes that resolved config to the re-audit.
        seen = []

        def revalidate_fn(config):
            seen.append(config)
            return _mia_result(member_mu=0.5 + 3 * config["a"])

        rows = validate_frontier(study, revalidate_fn)
        proxy_agreement(study, revalidate_fn, n_configs=3)
        assert seen and all(c["batch_size"] == 128 for c in seen)
        assert all(row["config"]["batch_size"] == 128 for row in rows)

    def test_proxy_agreement_returns_rho(self, tmp_path):
        study = _study_with_front(tmp_path)

        def revalidate_fn(params):
            return _mia_result(member_mu=0.5 + 3 * params["a"])

        out = proxy_agreement(study, revalidate_fn, n_configs=5)
        assert out["n_configs"] >= 2
        assert out["spearman_rho"] is None or -1.0 <= out["spearman_rho"] <= 1.0
