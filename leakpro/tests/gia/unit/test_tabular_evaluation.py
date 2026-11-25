#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the tabular evaluation stack.

Covers the three questions a tabular reconstruction metric has to answer
separately — per-value correctness, row identity, and whether the resulting
number beats a prior-only baseline — plus the schema validation that keeps
column indices honest.
"""

import numpy as np
import pytest
import torch


from leakpro.attacks.gia_attacks.modular.components.evaluators import (
    TabularEvaluator,
)
from leakpro.attacks.gia_attacks.modular.components.tabular.schema import TabularSchema


# ── TabularSchema validation ──────────────────────────────────────────────────

class TestTabularSchemaValidation:
    """Column indices are load-bearing in three modules; a bad layout must fail
    at construction rather than silently mislabel a metric later."""

    def test_overlapping_columns_rejected(self):
        with pytest.raises(ValueError, match="claimed by both"):
            TabularSchema(
                num_features=4,
                continuous_cols=[0, 1],
                categorical_col_groups=[(1, 2)],  # column 1 double-claimed
            )

    def test_out_of_range_column_rejected(self):
        with pytest.raises(ValueError, match="outside"):
            TabularSchema(num_features=3, continuous_cols=[0, 5])

    def test_both_encodings_rejected(self):
        with pytest.raises(ValueError, match="not both"):
            TabularSchema(
                num_features=3,
                continuous_cols=[0],
                categorical_col_groups=[(1, 2)],
                categorical_scalar_cols=[2],
            )

    def test_feature_names_length_must_match(self):
        with pytest.raises(ValueError, match="feature_names"):
            TabularSchema(num_features=3, continuous_cols=[0, 1, 2],
                          feature_names=["a", "b"])

    def test_valid_layout_accepted(self):
        schema = TabularSchema(
            num_features=5,
            continuous_cols=[0, 1],
            categorical_col_groups=[(2, 3)],
            feature_names=["age", "income", "c0", "c1", "c2"],
        )
        assert schema.num_scalar_features == 3
        assert schema.scalar_categorical_idx == [2]
        assert schema.scalar_feature_names == ["age", "income", "c0"]


class TestToScalarCodes:
    def test_one_hot_block_becomes_argmax(self):
        schema = TabularSchema(
            num_features=4, continuous_cols=[0], categorical_col_groups=[(1, 3)]
        )
        x = torch.tensor([[0.5, 0.1, 0.9, 0.2],
                          [1.5, 0.7, 0.2, 0.1]])
        codes = schema.to_scalar_codes(x)
        assert torch.equal(codes, torch.tensor([[0.5, 1.0], [1.5, 0.0]]))

    def test_scalar_schema_reorders_only(self):
        schema = TabularSchema(
            num_features=3, continuous_cols=[0, 2], categorical_scalar_cols=[1]
        )
        x = torch.tensor([[1.0, 7.0, 2.0]])
        # continuous first (cols 0, 2), then categorical (col 1)
        assert torch.equal(schema.to_scalar_codes(x), torch.tensor([[1.0, 2.0, 7.0]]))

    def test_wrong_width_rejected(self):
        schema = TabularSchema(num_features=3, continuous_cols=[0, 1, 2])
        with pytest.raises(ValueError, match="trailing dim"):
            schema.to_scalar_codes(torch.zeros(2, 4))


# ── TabularEvaluator ──────────────────────────────────────────────────────────

def _cont_schema(n: int = 3) -> TabularSchema:
    return TabularSchema(num_features=n, continuous_cols=list(range(n)))


class TestRowMatching:
    """A reconstructed batch is a set — scoring row i against row i reports
    near-chance accuracy for a perfect but permuted reconstruction."""

    def test_permuted_perfect_reconstruction_scores_100(self):
        schema = _cont_schema()
        true = torch.tensor([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0], [-3.0, -2.0, -1.0]])
        recon = true[[2, 0, 1]]  # same set, different order

        result = TabularEvaluator(schema, baseline_draws=1).evaluate(recon, true)
        assert result.metrics["reconstruction_accuracy"] == pytest.approx(100.0)
        assert result.metrics["exact_row_recovery"] == pytest.approx(1.0)

    def test_permutation_is_reported(self):
        schema = _cont_schema()
        true = torch.tensor([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0], [-3.0, -2.0, -1.0]])
        recon = true[[2, 0, 1]]
        result = TabularEvaluator(schema, baseline_draws=1).evaluate(recon, true)
        assert result.extras["permutation"] == [1, 2, 0]


class TestTolerance:
    """Continuous hits are within tol*sigma; categorical hits are exact."""

    def test_continuous_within_tolerance_counts_as_hit(self):
        schema = _cont_schema(1)
        true = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        sigma = true.std(dim=0, unbiased=False).item()
        # Nudge every value by just under the tolerance band.
        recon = true + 0.9 * 0.319 * sigma

        result = TabularEvaluator(schema, baseline_draws=1).evaluate(recon, true)
        assert result.metrics["reconstruction_accuracy"] == pytest.approx(100.0)

    def test_continuous_outside_tolerance_counts_as_miss(self):
        schema = _cont_schema(1)
        true = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        sigma = true.std(dim=0, unbiased=False).item()
        recon = true + 1.5 * 0.319 * sigma

        result = TabularEvaluator(schema, baseline_draws=1).evaluate(recon, true)
        assert result.metrics["reconstruction_accuracy"] == pytest.approx(0.0)

    def test_categorical_requires_exact_match(self):
        schema = TabularSchema(num_features=1, categorical_scalar_cols=[0])
        true = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        result = TabularEvaluator(schema, baseline_draws=1).evaluate(true + 0.01, true)
        assert result.metrics["reconstruction_accuracy"] == pytest.approx(0.0)


class TestBaselineCalibration:
    """The headline accuracy is meaningless without the prior-only comparison, and the prior must not be the truth."""

    def test_prior_only_reconstruction_has_near_zero_relative_gain(self):
        rng = np.random.default_rng(0)
        schema = _cont_schema(4)
        reference = torch.tensor(rng.normal(size=(500, 4)), dtype=torch.float32)
        true = torch.tensor(rng.normal(size=(60, 4)), dtype=torch.float32)
        # A "reconstruction" drawn from the same marginals but carrying no information about the victim's rows.
        recon = torch.tensor(rng.normal(size=(60, 4)), dtype=torch.float32)

        m = TabularEvaluator(schema, baseline_draws=5, reference_rows=reference).evaluate(recon, true).metrics
        assert abs(m["relative_gain"]) < 0.25, (
            f"A marginal-only reconstruction scored relative_gain={m['relative_gain']:.3f}; "
            "it should be near zero regardless of how high raw accuracy looks."
        )

    def test_perfect_reconstruction_has_relative_gain_one(self):
        rng = np.random.default_rng(1)
        schema = _cont_schema(4)
        reference = torch.tensor(rng.normal(size=(200, 4)), dtype=torch.float32)
        true = torch.tensor(rng.normal(size=(40, 4)), dtype=torch.float32)
        m = TabularEvaluator(schema, baseline_draws=3, reference_rows=reference).evaluate(true.clone(), true).metrics
        assert m["reconstruction_accuracy"] == pytest.approx(100.0)
        assert m["relative_gain"] == pytest.approx(1.0)

    def test_single_row_baseline_does_not_see_the_victim_row(self):
        """Resampling the victim's own batch made a 1-row baseline 100% and every relative gain 0."""
        rng = np.random.default_rng(2)
        schema = TabularSchema(num_features=4, continuous_cols=[0, 1], categorical_scalar_cols=[2, 3])
        reference = torch.cat(
            [torch.tensor(rng.normal(size=(300, 2)), dtype=torch.float32),
             torch.tensor(rng.integers(0, 5, size=(300, 2)), dtype=torch.float32)], dim=1)
        true = reference[:1].clone()
        wrong = true.clone()
        wrong[:, :2] += 10.0
        wrong[:, 2:] = (wrong[:, 2:] + 1) % 5
        ev = TabularEvaluator(schema, baseline_draws=3, reference_rows=reference, std_source="reference")
        perfect, bad = ev.evaluate(true.clone(), true).metrics, ev.evaluate(wrong, true).metrics
        assert perfect["baseline_accuracy"] < 100.0
        assert perfect["baseline_accuracy"] == pytest.approx(bad["baseline_accuracy"])
        assert perfect["relative_gain"] == pytest.approx(1.0)
        assert bad["relative_gain"] < 0.0

    def test_baseline_is_omitted_without_reference_rows(self):
        schema = _cont_schema(3)
        true = torch.randn(5, 3)
        m = TabularEvaluator(schema).evaluate(true.clone(), true).metrics
        assert "baseline_accuracy" not in m
        assert "relative_gain" not in m

    def test_evaluation_does_not_touch_global_numpy_rng(self):
        """The old implementation called np.random.seed(42) mid-metric, silently
        reseeding the whole process."""
        schema = _cont_schema(3)
        true = torch.randn(20, 3)
        np.random.seed(1234)
        before = np.random.random()

        np.random.seed(1234)
        TabularEvaluator(schema, baseline_draws=3).evaluate(torch.randn(20, 3), true)
        after = np.random.random()

        assert before == after, "Evaluation perturbed global numpy RNG state."


class TestRowCountMismatch:
    """N_hat != N is reported, never papered over by discarding random rows."""

    def test_fewer_recon_rows_reported_via_recall(self):
        schema = _cont_schema(2)
        true = torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        recon = true[:2].clone()  # recovered only half the dataset

        m = TabularEvaluator(schema, baseline_draws=1).evaluate(recon, true).metrics
        assert m["n_true"] == 4
        assert m["n_recon"] == 2
        assert m["n_matched"] == 2
        assert m["row_recall"] == pytest.approx(0.5)
        assert m["row_precision"] == pytest.approx(1.0)
        assert m["row_f1"] == pytest.approx(2 / 3)

    def test_more_recon_rows_reported_via_precision(self):
        schema = _cont_schema(2)
        true = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        recon = torch.tensor(
            [[0.0, 0.0], [10.0, 10.0], [99.0, 99.0], [77.0, 77.0]]
        )
        m = TabularEvaluator(schema, baseline_draws=1).evaluate(recon, true).metrics
        assert m["n_recon"] == 4
        assert m["row_recall"] == pytest.approx(1.0)
        assert m["row_precision"] == pytest.approx(0.5)

    def test_mismatch_result_is_deterministic(self):
        """Random row-discarding made the headline number partly a coin flip."""
        schema = _cont_schema(2)
        true = torch.randn(9, 2)
        recon = torch.randn(4, 2)
        ev = TabularEvaluator(schema, baseline_draws=1)
        first = ev.evaluate(recon, true).metrics["reconstruction_accuracy"]
        for _ in range(3):
            assert ev.evaluate(recon, true).metrics["reconstruction_accuracy"] == first


class TestPerColumnAndReidentification:
    def test_per_column_accuracy_uses_feature_names(self):
        schema = TabularSchema(
            num_features=2, continuous_cols=[0, 1], feature_names=["age", "income"]
        )
        true = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        recon = true.clone()
        recon[:, 1] += 100.0  # income destroyed, age perfect

        m = TabularEvaluator(schema, baseline_draws=1).evaluate(recon, true).metrics
        assert m["accuracy/age"] == pytest.approx(1.0)
        assert m["accuracy/income"] == pytest.approx(0.0)

    def test_reidentification_rate_is_one_for_exact_reconstruction(self):
        schema = _cont_schema(3)
        true = torch.randn(25, 3)
        m = TabularEvaluator(schema, baseline_draws=1).evaluate(true.clone(), true).metrics
        assert m["nn_reidentification_rate"] == pytest.approx(1.0)

    def test_reidentification_rate_is_zero_for_far_reconstruction(self):
        schema = _cont_schema(3)
        true = torch.randn(25, 3)
        m = TabularEvaluator(schema, baseline_draws=1).evaluate(true + 50.0, true).metrics
        assert m["nn_reidentification_rate"] == pytest.approx(0.0)
