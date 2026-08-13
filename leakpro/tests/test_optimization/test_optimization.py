#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the PET optimization module."""

import json

import numpy as np
import pytest

from leakpro.optimization import (
    AttackScores,
    Campaign,
    Knob,
    KnobSpace,
    clopper_pearson_ci,
    default_dpsgd_space,
    pareto_front,
    plot_frontier,
    tpr_at_fpr,
)


class TestKnobs:
    def test_bounds_respected_and_deterministic(self):
        space = default_dpsgd_space()
        a = space.sample_sobol(64, seed=7)
        b = space.sample_sobol(64, seed=7)
        assert a == b
        for config in a:
            assert 0.4 <= config["noise_multiplier"] <= 8.0
            assert 32 <= config["batch_size"] <= 1024
            assert isinstance(config["batch_size"], int)

    def test_log_scale_covers_decades(self):
        space = KnobSpace([Knob("lr", 1e-5, 1e-1, log_scale=True)])
        values = [c["lr"] for c in space.sample_sobol(256, seed=0)]
        assert min(values) < 1e-4 and max(values) > 1e-2

    def test_fix_and_narrow(self):
        space = default_dpsgd_space().fix("batch_size", 128).narrow("noise_multiplier", 1.0, 2.0)
        assert space.dim == 3
        for config in space.sample_sobol(16, seed=0):
            assert config["batch_size"] == 128
            assert 1.0 <= config["noise_multiplier"] <= 2.0

    def test_invalid_knobs_rejected(self):
        with pytest.raises(ValueError):
            Knob("x", 1.0, 1.0)
        with pytest.raises(ValueError):
            Knob("x", 0.0, 1.0, log_scale=True)
        with pytest.raises(ValueError):
            KnobSpace([Knob("x", 0, 1)], fixed={"x": 0.5})
        with pytest.raises(KeyError):
            default_dpsgd_space().fix("nope", 1.0)


class TestObjectives:
    def test_tpr_perfect_separation(self):
        scores = AttackScores(np.linspace(10, 20, 1000), np.linspace(0, 1, 1000))
        tpr, k, n = tpr_at_fpr(scores, 0.01)
        assert tpr == 1.0 and k == n == 1000

    def test_tpr_no_signal_is_near_fpr(self):
        rng = np.random.default_rng(0)
        scores = AttackScores(rng.normal(size=20000), rng.normal(size=20000))
        tpr, _, _ = tpr_at_fpr(scores, 0.01)
        assert 0.005 < tpr < 0.02

    def test_clopper_pearson_edges(self):
        low, high = clopper_pearson_ci(0, 100)
        assert low == 0.0 and 0.0 < high < 0.06
        low, high = clopper_pearson_ci(100, 100)
        assert 0.94 < low < 1.0 and high == 1.0
        with pytest.raises(ValueError):
            clopper_pearson_ci(5, 0)


def _fake_campaign(tmp_path, gate=None, seed=0):
    """Campaign over synthetic objectives: utility rises and leakage falls with noise."""

    def train(config):
        return config  # the "model" is just its config

    def utility(model):
        return 1.0 / (1.0 + model["noise_multiplier"])

    def attack(model, config):
        rng = np.random.default_rng(42)
        separation = 2.0 / (1.0 + model["noise_multiplier"])
        return AttackScores(rng.normal(separation, 1, 4000), rng.normal(0, 1, 4000))

    return Campaign(train, utility, attack, default_dpsgd_space(), tmp_path, utility_gate=gate, seed=seed)


class TestCampaign:
    def test_run_persists_and_resumes(self, tmp_path):
        records = _fake_campaign(tmp_path).run(8)
        assert len(records) == 8
        lines = (tmp_path / "evaluations.jsonl").read_text().strip().splitlines()
        assert len(lines) == 8
        assert all("attack_tpr" in json.loads(line) for line in lines)

        resumed = _fake_campaign(tmp_path)
        assert len(resumed.records) == 8
        assert len(resumed.run(8)) == 8  # nothing re-evaluated

    def test_utility_gate_skips_attack(self, tmp_path):
        records = _fake_campaign(tmp_path, gate=lambda u, _: u > 0.5).run(8)
        skipped = [r for r in records if r.get("attack_skipped")]
        attacked = [r for r in records if r.get("attack_tpr") is not None]
        assert skipped and attacked
        assert all(r["utility"] <= 0.5 for r in skipped)


class TestFrontier:
    def test_pareto_front_dominance(self):
        records = [
            {"utility": 0.9, "attack_tpr": 0.30},
            {"utility": 0.8, "attack_tpr": 0.10},
            {"utility": 0.7, "attack_tpr": 0.20},  # dominated by the one above
            {"utility": 0.5, "attack_tpr": 0.02},
            {"utility": 0.4, "attack_tpr": None},  # skipped, never on the front
        ]
        front = pareto_front(records)
        assert [r["utility"] for r in front] == [0.5, 0.8, 0.9]

    def test_plot_writes_file(self, tmp_path):
        records = _fake_campaign(tmp_path).run(8)
        out = plot_frontier(records, tmp_path / "frontier.png")
        assert out.exists() and out.stat().st_size > 0
