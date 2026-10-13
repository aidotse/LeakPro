#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for functional signals and the membership-direction registry.

Functional signals operate on cached logits/predictions (no model re-query), which is why
LiRA-style attacks use them. Here we cover the classification ``loss`` signal added for MS-LiRA
parity and the ``SIGNAL_MEMBERSHIP_DIRECTION`` orientation table.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

from leakpro.signals import functional


class TestFunctionalLoss:
    """``functional.loss`` should equal per-point cross-entropy from logits + integer labels."""

    def test_multiclass_matches_cross_entropy(self) -> None:
        """CE(z, y) = logsumexp(z) - z[y] for multi-class logits."""
        logits = np.array([[2.0, 1.0, 0.1], [0.0, 1.0, 2.0], [-1.0, 0.5, 0.5]])
        labels = np.array([0, 2, 1], dtype=np.int64)

        expected = logsumexp(logits, axis=1) - logits[np.arange(3), labels]
        np.testing.assert_allclose(functional.loss(logits, labels), expected, rtol=1e-6)

    def test_binary_single_logit_head(self) -> None:
        """A single-logit head is treated as binary: CE = softplus(-z) for y=1, softplus(z) for y=0."""
        logits = np.array([[2.0], [-1.5], [0.0]])
        labels = np.array([1, 0, 1], dtype=np.int64)

        z = logits.ravel()
        # -log sigmoid(z) for y=1; -log(1-sigmoid(z)) for y=0
        expected = np.where(labels == 1, np.logaddexp(0, -z), np.logaddexp(0, z))
        np.testing.assert_allclose(functional.loss(logits, labels), expected, rtol=1e-6)

    def test_lower_loss_for_confident_correct_prediction(self) -> None:
        """A confident correct prediction should have lower loss than an unconfident one."""
        confident = functional.loss(np.array([[10.0, 0.0]]), np.array([0], dtype=np.int64))
        unconfident = functional.loss(np.array([[0.1, 0.0]]), np.array([0], dtype=np.int64))
        assert confident[0] < unconfident[0]

    def test_requires_int64_labels(self) -> None:
        """Labels must be int64 (used for fancy indexing), mirroring rescaled_logits."""
        with pytest.raises(AssertionError):
            functional.loss(np.array([[1.0, 0.0]]), np.array([0], dtype=np.float64))


class TestRescaledSmape:
    """``functional.rescaled_smape`` must stay finite at both saturation points of SMAPE."""

    def test_perfect_prediction_is_finite(self) -> None:
        """An exactly reproduced series gives SMAPE 0; log(0) would be -inf."""
        series = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.all(np.isfinite(functional.rescaled_smape(series, series.copy())))

    def test_zero_target_is_finite(self) -> None:
        """A zero-valued target with a non-zero prediction gives SMAPE 1, the other singularity."""
        predictions = np.array([[1.0, 2.0, 3.0]])
        targets = np.zeros_like(predictions)
        assert np.all(np.isfinite(functional.rescaled_smape(predictions, targets)))

    def test_saturated_ends_are_symmetric(self) -> None:
        """The same epsilon on both ends puts the two extremes at +-log(1/eps)."""
        series = np.array([[1.0, 2.0, 3.0]])
        best = functional.rescaled_smape(series, series.copy())
        worst = functional.rescaled_smape(series, np.zeros_like(series))
        np.testing.assert_allclose(best, -worst, rtol=1e-6)

    def test_worse_prediction_scores_higher(self) -> None:
        """The transform is monotone in SMAPE: a worse prediction gives a larger value."""
        targets = np.array([[1.0, 2.0, 3.0]])
        close = functional.rescaled_smape(targets + 0.1, targets)
        far = functional.rescaled_smape(targets + 1.0, targets)
        assert close[0] < far[0]


class TestTs2vecEncoderSharing:
    """The ts2vec signal must encode every model with one shared encoder."""

    @staticmethod
    def _series(seed: int) -> tuple:
        rng = np.random.default_rng(seed)
        targets = rng.normal(size=(8, 16, 1)).astype(np.float32)
        predictions = targets + rng.normal(scale=0.1, size=targets.shape).astype(np.float32)
        return predictions, targets

    def test_a_bound_encoder_makes_repeated_calls_identical(self) -> None:
        """Two calls on the same input must agree, or the value is not comparable across models."""
        from ts2vec import TS2Vec
        predictions, targets = self._series(0)
        encoder = TS2Vec(input_dims=1, device="cpu", batch_size=4)
        encoder.fit(targets)

        first = functional.ts2vec(predictions, targets, batch_size=4, encoder=encoder)
        second = functional.ts2vec(predictions, targets, batch_size=4, encoder=encoder)
        np.testing.assert_array_equal(first, second)

    def test_without_an_encoder_each_call_fits_its_own(self) -> None:
        """Documents why the encoder must be passed: unbound calls disagree on identical input."""
        predictions, targets = self._series(1)
        first = functional.ts2vec(predictions, targets, batch_size=4)
        second = functional.ts2vec(predictions, targets, batch_size=4)
        assert not np.array_equal(first, second)

    def test_bind_fits_one_encoder_on_the_shadow_population(self, tmp_path: object) -> None:
        """The bound signal must reuse its encoder, and fit it on the shadow population."""
        import types

        from leakpro.signals.utils.get_TS2Vec import bind_ts2vec_encoder
        predictions, targets = self._series(2)
        population = np.concatenate([targets, targets], axis=0)  # audited points plus extra
        handler = types.SimpleNamespace(
            configs=types.SimpleNamespace(audit=types.SimpleNamespace(output_dir=str(tmp_path))),
            population=types.SimpleNamespace(targets=population),
        )

        bound = bind_ts2vec_encoder(functional.ts2vec, handler, np.arange(len(targets)), batch_size=4)
        assert bound is not functional.ts2vec
        np.testing.assert_array_equal(bound(predictions, targets), bound(predictions, targets))

    def test_bind_leaves_other_signals_untouched(self) -> None:
        """Only ts2vec carries a fitted artefact, so nothing else may be wrapped."""
        from leakpro.signals.utils.get_TS2Vec import bind_ts2vec_encoder
        for signal_fn in [functional.mse, functional.dtw, functional.rescaled_logits]:
            assert bind_ts2vec_encoder(signal_fn, None, None) is signal_fn


class TestSignalMembershipDirection:
    """The orientation table encodes whether higher (+1) or lower (-1) means membership."""

    def test_confidence_signals_are_higher_is_member(self) -> None:
        """Confidence-style signals: a higher value indicates membership (+1)."""
        assert functional.SIGNAL_MEMBERSHIP_DIRECTION["rescaled_logits"] == +1
        assert functional.SIGNAL_MEMBERSHIP_DIRECTION["logits"] == +1

    def test_error_and_distance_signals_are_lower_is_member(self) -> None:
        """Error/distance signals: a lower value indicates membership (-1)."""
        for name in ["loss", "mse", "mae", "smape", "rescaled_smape",
                     "seasonality", "trend", "ts2vec", "dtw", "msm"]:
            assert functional.SIGNAL_MEMBERSHIP_DIRECTION[name] == -1, name

    def test_every_direction_is_plus_or_minus_one(self) -> None:
        """Directions are strictly +1 or -1 (used as a multiplicative sign)."""
        assert set(functional.SIGNAL_MEMBERSHIP_DIRECTION.values()) <= {+1, -1}


class TestGet:
    """``functional.get`` maps a config signal name to its signal function."""

    def test_class_names_map_to_functional(self) -> None:
        """Class-style config names resolve to their functional equivalents."""
        assert functional.get("ModelRescaledLogits") is functional.rescaled_logits
        assert functional.get("ModelLoss") is functional.loss
        assert functional.get("MSE") is functional.mse

    def test_functional_names_resolve_directly(self) -> None:
        """Functional names resolve to the same-named function."""
        assert functional.get("rescaled_logits") is functional.rescaled_logits
        assert functional.get("dtw") is functional.dtw

    def test_unknown_signal_raises(self) -> None:
        """An unknown signal raises rather than returning None the way dict.get would."""
        with pytest.raises(ValueError, match="Unknown signal"):
            functional.get("NotASignal")

    def test_module_attribute_is_not_a_signal(self) -> None:
        """Module-level imports are not resolvable as signals, unlike a getattr-based lookup."""
        for name in ["np", "torch", "Parallel", "SIGNAL_MEMBERSHIP_DIRECTION"]:
            with pytest.raises(ValueError, match="Unknown signal"):
                functional.get(name)

    def test_alias_wins_over_a_same_named_import(self) -> None:
        """``TS2Vec`` names the signal, not the imported TS2Vec model class it collides with."""
        assert functional.get("TS2Vec") is functional.ts2vec

    def test_error_tells_a_contributor_how_to_register_a_signal(self) -> None:
        """The error must name both registries: a new signal is unusable until it is in both."""
        with pytest.raises(ValueError) as excinfo:
            functional.get("kurtosis")
        message = str(excinfo.value)
        assert "SIGNAL_FUNCTIONS" in message
        assert "SIGNAL_MEMBERSHIP_DIRECTION" in message


class TestDirection:
    """``functional.direction`` resolves the same names as ``get`` to an orientation."""

    def test_class_names(self) -> None:
        """Class-style config names resolve to a direction via the alias mapping."""
        assert functional.direction("ModelRescaledLogits") == +1
        assert functional.direction("ModelLoss") == -1
        assert functional.direction("MSE") == -1

    def test_functional_names(self) -> None:
        """Functional names resolve directly."""
        assert functional.direction("rescaled_logits") == +1
        assert functional.direction("mse") == -1

    def test_unknown_signal_raises(self) -> None:
        """An undeclared signal raises rather than silently defaulting a direction."""
        with pytest.raises(ValueError, match="Unknown signal"):
            functional.direction("NotASignal")


class TestSignalRegistries:
    """Every usable signal declares both a function and a membership direction."""

    def test_registries_declare_the_same_signals(self) -> None:
        """A signal missing from either registry is a half-registered signal."""
        assert functional.SIGNAL_FUNCTIONS.keys() == functional.SIGNAL_MEMBERSHIP_DIRECTION.keys()

    def test_aliases_point_at_registered_signals(self) -> None:
        """Every class-style alias resolves to a signal that exists."""
        for alias, name in functional.LEGACY_SIGNAL_NAMES.items():
            assert name in functional.SIGNAL_FUNCTIONS, alias

    def test_registered_functions_are_named_consistently(self) -> None:
        """The registry key matches the function it points at, so configs and code agree."""
        for name, fn in functional.SIGNAL_FUNCTIONS.items():
            assert fn.__name__ == name
