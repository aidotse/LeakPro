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
    proxy_agreement,
    resolution_warning,
    tpr_at_fpr,
    validate_frontier,
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

    class FakeModel(dict):
        campaign_extras = None

    def train(config):
        model = FakeModel(config)  # the "model" is just its config
        model.campaign_extras = {"epsilon": 1.0 / config["noise_multiplier"]}
        return model

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
        lines = [json.loads(line) for line in (tmp_path / "evaluations.jsonl").read_text().strip().splitlines()]
        assert len(lines) == 8
        assert all("attack_tpr" in line for line in lines)
        assert all("epsilon" in line for line in lines)  # campaign_extras recorded

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
        assert out.exists()
        assert out.stat().st_size > 0


def _fake_revalidate(separation_of):
    """Stronger attack stand-in: bigger audit set, same underlying separation."""

    def revalidate_fn(config):
        rng = np.random.default_rng(7)
        sep = separation_of(config)
        return AttackScores(rng.normal(sep, 1, 20000), rng.normal(0, 1, 20000))

    return revalidate_fn


class TestValidation:
    def test_resolution_warning_fires_only_when_underpowered(self):
        assert resolution_warning(2000, 0.001) is not None  # 2 expected events
        assert resolution_warning(20000, 0.001) is None  # 20 expected events

    def test_validate_frontier_reports_each_fpr(self, tmp_path):
        records = _fake_campaign(tmp_path).run(8)
        validated = validate_frontier(
            records,
            _fake_revalidate(lambda c: 2.0 / (1.0 + c["noise_multiplier"])),
            report_fprs=(0.001, 0.01),
        )
        assert validated
        assert len(validated) == len(pareto_front(records))
        for record in validated:
            block = record["validation"]
            assert block["n_nonmembers"] == 20000
            for key in ("tpr_at_0.001", "tpr_at_0.01"):
                assert 0.0 <= block[key]["tpr"] <= 1.0
                assert block[key]["ci95"][0] <= block[key]["tpr"] <= block[key]["ci95"][1]
            assert block["tpr_at_0.001"]["warning"] is None  # 20k nonmembers is enough

    def test_proxy_agreement_detects_agreement(self, tmp_path):
        records = _fake_campaign(tmp_path).run(12)
        result = proxy_agreement(
            records,
            _fake_revalidate(lambda c: 2.0 / (1.0 + c["noise_multiplier"])),
            n_configs=6,
        )
        assert result["spearman_rho"] > 0.7  # same mechanism drives both FPR levels
        assert len(result["pairs"]) == 6

    def test_proxy_agreement_detects_disagreement(self, tmp_path):
        records = _fake_campaign(tmp_path).run(12)
        # Tail behaviour inverted relative to the proxy: ranking must not survive.
        result = proxy_agreement(
            records,
            _fake_revalidate(lambda c: c["noise_multiplier"] / 4.0),
            n_configs=6,
        )
        assert result["spearman_rho"] < 0.0


class TestTieSafety:
    """tpr_at_fpr must never realize a higher FPR than requested — DP-SGD
    saturates outputs, so tied nonmember blocks are the normal case, not an edge."""

    @staticmethod
    def _realized_fpr(nonmembers, tpr_threshold_probe_members, fpr):
        # Recompute what FPR the chosen operating point actually admits, by
        # finding the member-counting rule's threshold implicitly: any member
        # score x is counted iff tpr counts it; probe with the nonmembers.
        tpr, _, n = tpr_at_fpr(AttackScores(np.asarray(nonmembers, float),
                                            np.asarray(nonmembers, float)), fpr)
        # Using the nonmembers as members: TPR == realized FPR by construction.
        return tpr

    def test_all_tied_nonmembers_do_not_blow_the_budget(self):
        nm = np.zeros(1000)
        m = np.zeros(1000)
        tpr, k, n = tpr_at_fpr(AttackScores(m, nm), 0.01)
        assert tpr == 0.0  # counting the tied block would mean 100% FPR
        assert self._realized_fpr(nm, m, 0.01) <= 0.01

    def test_tie_block_crossing_the_boundary_is_excluded(self):
        nm = np.concatenate([np.ones(500), np.zeros(500)])
        m = np.ones(1000)
        tpr, _, _ = tpr_at_fpr(AttackScores(m, nm), 0.01)
        # Admitting the 500-strong tied block would realize 50% FPR at a 1% budget.
        assert tpr == 0.0
        assert self._realized_fpr(nm, m, 0.01) <= 0.01

    def test_partial_tie_falls_back_to_next_distinct_value(self):
        # 5 nonmembers at 2.0, 995 below: the block fits inside a 1% budget.
        nm = np.concatenate([np.full(5, 2.0), np.linspace(-1, 1, 995)])
        m = np.full(100, 2.0)
        tpr, _, _ = tpr_at_fpr(AttackScores(m, nm), 0.01)
        assert tpr == 1.0  # threshold sits at 2.0; realized FPR = 5/1000 <= 1%

    def test_distinct_scores_match_the_classic_rule(self):
        rng = np.random.default_rng(0)
        nm = rng.normal(size=2000)  # continuous -> ties have measure zero
        m = rng.normal(1.0, 1.0, size=2000)
        tpr, _, _ = tpr_at_fpr(AttackScores(m, nm), 0.01)
        n_admit = int(np.floor(0.01 * nm.size))
        classic = np.mean(m >= np.sort(nm)[::-1][n_admit - 1])
        assert tpr == pytest.approx(classic)


class TestCampaignRobustness:
    def test_resume_with_different_identity_refuses(self, tmp_path):
        _fake_campaign(tmp_path, seed=0).run(4)
        with pytest.raises(ValueError, match="different seed"):
            _fake_campaign(tmp_path, seed=1)

    def test_failing_config_is_recorded_and_does_not_kill_the_sweep(self, tmp_path):
        calls = {"n": 0}

        def flaky_train(config):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            model = type("M", (dict,), {"campaign_extras": None})(config)
            model.campaign_extras = {"epsilon": 1.0}
            return model

        def utility(model):
            return 0.5

        def attack(model, config):
            rng = np.random.default_rng(0)
            return AttackScores(rng.normal(1, 1, 500), rng.normal(0, 1, 500))

        campaign = Campaign(flaky_train, utility, attack, default_dpsgd_space(), tmp_path)
        records = campaign.run(4)
        errored = [r for r in records if "error" in r]
        assert len(records) == 4
        assert len(errored) == 1 and "boom" in errored[0]["error"]

        # Resume retries the errored config instead of livelocking or skipping it.
        resumed = Campaign(flaky_train, utility, attack, default_dpsgd_space(), tmp_path)
        final = resumed.run(4)
        ok_indices = {r["index"] for r in final if "error" not in r}
        assert ok_indices == {0, 1, 2, 3}

    def test_anchor_runs_under_reserved_negative_index(self, tmp_path):
        anchor = {"noise_multiplier": 0.0, "max_grad_norm": 1.0, "learning_rate": 0.05, "batch_size": 128}

        def train(config):
            model = type("M", (dict,), {"campaign_extras": None})(config)
            eps = float("inf") if config["noise_multiplier"] == 0 else 1.0
            model.campaign_extras = {"epsilon": eps}
            return model

        def utility(model):
            return 0.9

        def attack(model, config):
            rng = np.random.default_rng(1)
            return AttackScores(rng.normal(1, 1, 500), rng.normal(0, 1, 500))

        campaign = Campaign(train, utility, attack, default_dpsgd_space(), tmp_path)
        records = campaign.run(3, anchors=[anchor])
        assert {r["index"] for r in records} == {-1, 0, 1, 2}
        anchor_record = next(r for r in records if r["index"] == -1)
        # epsilon = inf is stored as None: bare Infinity is not JSON and browsers reject it.
        assert anchor_record["epsilon"] is None
        raw = (tmp_path / "evaluations.jsonl").read_text()
        assert "Infinity" not in raw and json.loads(raw.splitlines()[0]) is not None


class TestProxyAgreementNaN:
    def test_constant_rankings_report_inconclusive_not_a_number(self, tmp_path):
        records = _fake_campaign(tmp_path).run(6)

        def constant_revalidate(config):
            # Target-FPR re-attack that cannot resolve anything: all scores tied.
            return AttackScores(np.zeros(500), np.zeros(500))

        out = proxy_agreement(records, constant_revalidate, n_configs=4)
        assert out["spearman_rho"] is None
        assert out["p_value"] is None
        assert "INCONCLUSIVE" in (out["resolution_warning"] or "")
