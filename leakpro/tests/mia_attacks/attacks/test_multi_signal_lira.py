#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the multi-signal LiRA attack (AttackMSLiRA).

The end-to-end suite only exercises the *default* MS-LiRA config (a single
``ModelRescaledLogits`` signal, offline, on image data). These tests cover the
path that actually distinguishes MS-LiRA from plain LiRA: combining more than
one signal into a single joint membership score.

The shared ``image_handler`` fixture uses population == audit, so the offline
attack cannot run on it (MS-LiRA, like RMIA, requires spare auxiliary data for
the shadow models). Offline single-signal coverage lives in the end-to-end
suite; here we exercise the online attack, where multiple signals are combined.
"""

import numpy as np
import pytest
from scipy.stats import norm

from leakpro.attacks.mia_attacks.multi_signal_lira import (
    AttackMSLiRA,
    _resolve_signal_fn,
    _signal_membership_direction,
)
from leakpro.signals import functional
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.tests.constants import get_shadow_model_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler


def _ms_config(**overrides: object) -> dict:
    """Build an online MS-LiRA config dict with sane test defaults."""
    config = {
        "signals": ["ModelRescaledLogits", "ModelLoss"],
        "num_shadow_models": 4,
        "training_data_fraction": 0.5,
        "online": True,
        "var_calculation": "carlini",
    }
    config.update(overrides)
    return config


def _ensure_shadow_handler(image_handler: ImageInputHandler) -> None:
    image_handler.configs.shadow_model = get_shadow_model_config()
    if ShadowModelHandler.is_created() is False:
        ShadowModelHandler(image_handler)


def test_mslira_setup(image_handler: ImageInputHandler) -> None:
    """Should instantiate one signal object per configured signal name."""
    attack = AttackMSLiRA(image_handler, _ms_config())

    assert attack is not None
    assert attack.target_model is not None
    assert attack.signals == ["ModelRescaledLogits", "ModelLoss"]
    assert len(attack.signal_fns) == 2

    description = attack.description()
    assert len(description) == 4


def test_mslira_offline_requires_aux_data(image_handler: ImageInputHandler) -> None:
    """Offline construction must fail when there is no data left for shadow models.

    The fixture has population == audit, so an offline attack would leave no
    auxiliary data. MS-LiRA guards against this at construction time.
    """
    with pytest.raises(ValueError, match="no data left for the shadow models"):
        AttackMSLiRA(image_handler, _ms_config(online=False))


def test_mslira_prepare_stacks_signals(image_handler: ImageInputHandler) -> None:
    """prepare_attack should stack per-signal values on the last axis."""
    _ensure_shadow_handler(image_handler)
    attack = AttackMSLiRA(image_handler, _ms_config())
    attack.prepare_attack()

    n_signals = len(attack.signals)
    n_models = attack.num_shadow_models
    n_points = attack.shadow_models_signals.shape[0]

    assert attack.shadow_models_signals.shape == (n_points, n_models, n_signals)
    assert attack.target_model_signals.shape == (n_points, n_signals)
    assert n_points > 0


def test_mslira_online_multi_signal_runs(image_handler: ImageInputHandler) -> None:
    """Online multi-signal attack should produce finite, correctly-sized scores."""
    _ensure_shadow_handler(image_handler)
    attack = AttackMSLiRA(image_handler, _ms_config())
    attack.prepare_attack()

    result = attack.run_attack()

    n_points = len(attack.audit_data_indices)
    assert len(attack.in_member_signals) + len(attack.out_member_signals) == n_points
    assert not np.any(np.isnan(attack.in_member_signals))
    assert not np.any(np.isnan(attack.out_member_signals))
    assert isinstance(result, MIAResult)


def test_mslira_online_single_signal_runs(image_handler: ImageInputHandler) -> None:
    """A single-signal online run is the MS-LiRA analogue of plain online LiRA."""
    _ensure_shadow_handler(image_handler)
    attack = AttackMSLiRA(image_handler, _ms_config(signals=["ModelRescaledLogits"]))
    attack.prepare_attack()

    result = attack.run_attack()

    assert attack.target_model_signals.shape == (len(attack.audit_data_indices), 1)
    assert not np.any(np.isnan(attack.in_member_signals))
    assert not np.any(np.isnan(attack.out_member_signals))
    assert isinstance(result, MIAResult)


def test_mslira_resolves_functional_signals(image_handler: ImageInputHandler) -> None:
    """Signals must resolve to functions in leakpro.signals.functional, not class instances.

    This is the contract of the Part 1 migration: MS-LiRA scores cached logits with functional
    signals rather than re-querying models through Signal objects.
    """
    attack = AttackMSLiRA(image_handler, _ms_config(signals=["ModelRescaledLogits", "ModelLoss"]))
    assert attack.signal_fns == [functional.rescaled_logits, functional.loss]
    assert all(callable(s) for s in attack.signal_fns)


def test_mslira_does_not_requery_models_per_signal(
    image_handler: ImageInputHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare_attack must read each model's logits from cache exactly once, regardless of signal count.

    The old class-based implementation re-ran every model once per signal (issue #390). The
    functional path loads each model's logits a single time and applies all signals to the cached
    array, so load_logits is called exactly (num_shadow_models + 1) times — independent of how many
    signals are configured.
    """
    _ensure_shadow_handler(image_handler)
    attack = AttackMSLiRA(image_handler, _ms_config(signals=["ModelRescaledLogits", "ModelLoss"]))

    # ShadowModelHandler is a singleton factory, so every ShadowModelHandler() call returns the
    # same instance; shadowing load_logits on that instance counts every load in prepare_attack.
    handler_instance = ShadowModelHandler()
    original_load_logits = handler_instance.load_logits
    call_count = {"n": 0}

    def counting_load_logits(*args: object, **kwargs: object) -> np.ndarray:
        call_count["n"] += 1
        return original_load_logits(*args, **kwargs)

    monkeypatch.setattr(handler_instance, "load_logits", counting_load_logits)
    attack.prepare_attack()

    # One load for the target model + one per shadow model. The two signals add no extra loads.
    assert call_count["n"] == attack.num_shadow_models + 1


def test_mslira_undeclared_signal_direction_fails(image_handler: ImageInputHandler) -> None:
    """A signal with no membership-direction entry must fail loudly at construction.

    HopSkipJumpDistance is a registered signal but has no functional twin / direction, so it is
    not a valid MS-LiRA signal and should be rejected before any models are trained.
    """
    with pytest.raises(ValueError, match="No membership direction defined"):
        AttackMSLiRA(image_handler, _ms_config(signals=["HopSkipJumpDistance"]))


# ---------------------------------------------------------------------------
# Scoring formulas (pure unit tests — the offline path cannot run on the
# population==audit fixture, so the math is exercised via the static helpers).
# ---------------------------------------------------------------------------

class TestSignalDirectionResolution:
    """_signal_membership_direction accepts both class-style and functional names."""

    def test_class_names(self) -> None:
        """Class-style config names resolve to a direction via the functional mapping."""
        assert _signal_membership_direction("ModelRescaledLogits") == +1
        assert _signal_membership_direction("ModelLoss") == -1
        assert _signal_membership_direction("MSE") == -1

    def test_functional_names(self) -> None:
        """Functional names resolve directly."""
        assert _signal_membership_direction("rescaled_logits") == +1
        assert _signal_membership_direction("mse") == -1

    def test_unknown_signal_raises(self) -> None:
        """An undeclared signal raises rather than silently defaulting a direction."""
        with pytest.raises(ValueError, match="No membership direction defined"):
            _signal_membership_direction("NotASignal")


class TestResolveSignalFn:
    """_resolve_signal_fn maps config names to leakpro.signals.functional functions."""

    def test_class_names_map_to_functional(self) -> None:
        """Class-style config names resolve to their functional equivalents."""
        assert _resolve_signal_fn("ModelRescaledLogits") is functional.rescaled_logits
        assert _resolve_signal_fn("ModelLoss") is functional.loss
        assert _resolve_signal_fn("MSE") is functional.mse

    def test_functional_names_resolve_directly(self) -> None:
        """Functional names resolve to the same-named function."""
        assert _resolve_signal_fn("rescaled_logits") is functional.rescaled_logits
        assert _resolve_signal_fn("dtw") is functional.dtw

    def test_unknown_signal_raises(self) -> None:
        """An unknown signal raises rather than silently returning a non-signal attribute."""
        with pytest.raises(ValueError, match="Unknown signal"):
            _resolve_signal_fn("NotASignal")


class TestOfflineScore:
    """Offline score = one-sided multivariate Gaussian tail (paper Eq. 2)."""

    def test_single_signal_reduces_to_lira_logcdf(self) -> None:
        """For one signal the offline score is exactly LiRA's offline logcdf (Carlini Eq. 4)."""
        target = np.array([1.3])
        out_mean = np.array([0.5])
        out_std = np.array([0.8])
        eps = 1e-30

        score = AttackMSLiRA._offline_score(target, out_mean, out_std, eps)
        lira_offline = norm.logcdf(target[0], out_mean[0], out_std[0] + eps)  # == lira.py:222
        assert score == pytest.approx(lira_offline)

    def test_multi_signal_sums_per_signal_logcdf(self) -> None:
        """With diagonal covariance the joint tail prob is the sum of per-signal logcdf."""
        target = np.array([1.3, -0.2, 0.7])
        out_means = np.array([0.5, 0.0, 1.0])
        out_stds = np.array([0.8, 1.2, 0.5])
        eps = 1e-30

        score = AttackMSLiRA._offline_score(target, out_means, out_stds, eps)
        expected = norm.logcdf(target, out_means, out_stds + eps).sum()
        assert score == pytest.approx(expected)

    def test_is_monotone_increasing_in_target(self) -> None:
        """Higher (oriented) signal => more member-like => higher score."""
        out_mean, out_std, eps = np.array([0.0]), np.array([1.0]), 1e-30
        low = AttackMSLiRA._offline_score(np.array([-1.0]), out_mean, out_std, eps)
        high = AttackMSLiRA._offline_score(np.array([1.0]), out_mean, out_std, eps)
        assert high > low


class TestOnlineScore:
    """Online score = summed per-signal log-likelihood ratio (paper Eq. 1)."""

    def test_matches_sum_of_log_ratios(self) -> None:
        """Online score equals the summed per-signal (logpdf_in - logpdf_out)."""
        target = np.array([1.3, -0.2])
        in_means, in_stds = np.array([1.0, 0.0]), np.array([0.5, 1.0])
        out_means, out_stds = np.array([0.0, 0.5]), np.array([0.8, 1.2])
        eps = 1e-30

        score = AttackMSLiRA._online_score(target, in_means, in_stds, out_means, out_stds, eps)
        expected = (norm.logpdf(target, in_means, in_stds + eps)
                    - norm.logpdf(target, out_means, out_stds + eps)).sum()
        assert score == pytest.approx(expected)

    def test_score_invariant_to_signal_orientation(self) -> None:
        """Negating a signal (orientation flip) leaves the online likelihood ratio unchanged."""
        target = np.array([1.3])
        in_means, in_stds = np.array([1.0]), np.array([0.5])
        out_means, out_stds = np.array([0.0]), np.array([0.8])
        eps = 1e-30

        base = AttackMSLiRA._online_score(target, in_means, in_stds, out_means, out_stds, eps)
        flipped = AttackMSLiRA._online_score(-target, -in_means, in_stds, -out_means, out_stds, eps)
        assert base == pytest.approx(flipped)
