#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#

from types import SimpleNamespace

import numpy as np
import pytest
from dotmap import DotMap
from pydantic import ValidationError
from torch import nn

from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.attacks.utils.utils import (
    gaussian_residual_probability,
    huber_residual_probability,
    laplace_residual_probability,
    softmax_logits,
)
from leakpro.reporting.mia_result import MIAResult
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.tests.constants import get_audit_config, get_shadow_model_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler


def test_rmia_setup(image_handler:ImageInputHandler) -> None:
    """Test the initialization of LiRA."""
    
    # Try default params
    default_params = {
            "num_shadow_models": 1,
            "offline_a": 0.33,
            "gamma": 2.0,
            "temperature": 2.0,
            "training_data_fraction": 0.5,
            "online": False
        }
    rmia_params = None
    rmia_obj = AttackRMIA(image_handler, rmia_params) # create RMIA with default params
    # Ensure default parameters are set
    for key in default_params:
        assert getattr(rmia_obj, key) == default_params[key]
    
    # Try custom params
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    rmia_params = DotMap({k: v for k, v in audit_config.attack_list[1].items() if k != "attack"})
    rmia_obj = AttackRMIA(image_handler, rmia_params)

    assert rmia_obj is not None
    assert rmia_obj.target_model is not None
    assert rmia_obj.online == rmia_params.online
    assert rmia_obj.num_shadow_models == rmia_params.num_shadow_models
    assert rmia_obj.training_data_fraction == rmia_params.training_data_fraction

    description = rmia_obj.description()
    assert len(description) == 4

def test_rmia_prepare_online_attack(image_handler:ImageInputHandler) -> None:
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    rmia_params = DotMap({k: v for k, v in audit_config.attack_list[1].items() if k != "attack"})
    rmia_params.online = True

    image_handler.configs.shadow_model = get_shadow_model_config()
    rmia_obj = AttackRMIA(image_handler, rmia_params)

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    rmia_obj.prepare_attack()

    # ensure correct number of shadow models are read
    assert len(rmia_obj.shadow_models) == rmia_params.num_shadow_models
    # ensure the attack data indices correspond to the entire population pool
    assert sorted(rmia_obj.attack_data_indices) == list(range(image_handler.population_size))


def test_rmia_prepare_offline_attack(image_handler:ImageInputHandler) -> None:
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    rmia_params = DotMap({k: v for k, v in audit_config.attack_list[1].items() if k != "attack"})
    rmia_params.online = False

    image_handler.configs.shadow_model = get_shadow_model_config()
    rmia_obj = AttackRMIA(image_handler, rmia_params)

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    rmia_obj.prepare_attack()

    # ensure correct number of shadow models are read
    assert len(rmia_obj.shadow_models) == rmia_params.num_shadow_models
    # ensure the attack data indices correspond to the correct pool
    assert sorted(rmia_obj.attack_data_indices) == sorted(range(image_handler.population_size))

    # Check that the filtering of the attack data is correct (this is done after shadow models are created)
    n_attack_points = len(rmia_obj.attack_data_indices) * rmia_params.attack_data_fraction
    assert len(rmia_obj.ratio_z.squeeze()) == n_attack_points
    assert rmia_obj.ratio_z.all() >= 0.0
    assert rmia_obj.ratio_z.all() <= 1.0


def test_rmia_online_attack(image_handler:ImageInputHandler):
    # Set up for testing
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    rmia_params = DotMap({k: v for k, v in audit_config.attack_list[1].items() if k != "attack"})
    rmia_params.online = True
    image_handler.configs.shadow_model = get_shadow_model_config()
    rmia_obj = AttackRMIA(image_handler, rmia_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    rmia_obj.prepare_attack()
    # Attack points (not shadow models) are not sampled from train/test data
    n_attack_points = len(rmia_obj.attack_data_indices)

    # Test attack
    rmia_obj.gamma = 1.0 # model is not trained so no strong signals
    rmia_obj.run_attack()
    
    n_audit_points = len(rmia_obj.audit_dataset["data"]) # Total number of points that can be audited
    assert len(rmia_obj.in_member_signals)+len(rmia_obj.out_member_signals) <= n_audit_points # Not all points will have both in and out models
    assert not np.any(np.isnan(rmia_obj.in_member_signals))
    assert not np.any(np.isnan(rmia_obj.out_member_signals))
    assert not np.any(np.isinf(rmia_obj.in_member_signals))
    assert not np.any(np.isinf(rmia_obj.out_member_signals))



def test_rmia_offline_attack(image_handler:ImageInputHandler):
    # Set up for testing
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    rmia_params = DotMap({k: v for k, v in audit_config.attack_list[1].items() if k != "attack"})
    rmia_params.online = False
    image_handler.configs.shadow_model = get_shadow_model_config()
    rmia_obj = AttackRMIA(image_handler, rmia_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    rmia_obj.prepare_attack()

    # Test attack
    rmia_obj.gamma = 1.0 # model is not trained so no strong signals
    rmia_result = rmia_obj.run_attack()
    n_attack_points = len(rmia_obj.attack_data_indices) * rmia_params.attack_data_fraction

    assert len(rmia_obj.ratio_z.squeeze()) == n_attack_points
    assert len(rmia_obj.in_member_signals)+len(rmia_obj.out_member_signals) == rmia_obj.audit_dataset["data"].shape[0]
    assert not np.any(np.isnan(rmia_obj.in_member_signals))
    assert not np.any(np.isnan(rmia_obj.out_member_signals))
    assert not np.any(np.isinf(rmia_obj.in_member_signals))
    assert not np.any(np.isinf(rmia_obj.out_member_signals))

    assert rmia_result is not None
    assert isinstance(rmia_result, MIAResult)
    
# ---------------------------------------------------------------------------
# Likelihood-family resolution and the residual (regression/forecasting) signal path
# ---------------------------------------------------------------------------

def test_rmia_should_resolve_classification_task_when_criterion_is_cross_entropy(image_handler:ImageInputHandler) -> None:
    rmia_obj = AttackRMIA(image_handler, None)

    assert isinstance(image_handler.get_criterion(), nn.CrossEntropyLoss)
    assert rmia_obj.is_regression is False
    assert rmia_obj.likelihood_family is None


@pytest.mark.parametrize(
    ("criterion", "expected_family", "expected_delta"),
    [
        (nn.MSELoss(), "gaussian", None),
        (nn.L1Loss(), "laplace", None),
        (nn.SmoothL1Loss(beta=0.5), "huber", 0.5),
        (nn.HuberLoss(delta=2.0), "huber", 2.0),
    ],
    ids=["mse-gaussian", "l1-laplace", "smooth_l1-huber", "huber-huber"],
)
def test_rmia_should_resolve_likelihood_family_from_criterion(
        image_handler:ImageInputHandler, criterion, expected_family, expected_delta) -> None:
    original_criterion = image_handler._criterion
    image_handler._criterion = criterion
    try:
        rmia_obj = AttackRMIA(image_handler, None)
        assert rmia_obj.is_regression is True
        assert rmia_obj.likelihood_family == expected_family
        assert rmia_obj._huber_delta == expected_delta
        assert rmia_obj._scale is None  # unresolved until prepare_attack
    finally:
        image_handler._criterion = original_criterion


def test_rmia_config_should_reject_signal_field() -> None:
    # 'signal' used to be a silently-ignored config field; it must now fail loudly.
    with pytest.raises(ValidationError):
        AttackRMIA.AttackConfig(signal="mse")


def test_signal_probability_should_match_softmax_indexing_when_classification() -> None:
    stub = SimpleNamespace(is_regression=False, likelihood_family=None, temperature=2.0, _scale=None, _huber_delta=None)
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 3))
    labels = np.array([0, 2, 1, 1])

    probabilities = AttackRMIA._signal_probability(stub, logits, labels)

    expected = softmax_logits(logits, 2.0)[np.arange(4), labels]
    assert np.allclose(probabilities, expected)


def test_signal_probability_should_use_gaussian_residual_when_criterion_is_mse() -> None:
    stub = SimpleNamespace(is_regression=True, likelihood_family="gaussian", temperature=2.0, _scale=0.5, _huber_delta=None)
    rng = np.random.default_rng(1)
    forecasts = rng.normal(size=(6, 8))      # (n_points, horizon)
    true_horizons = rng.normal(size=(6, 8))  # float labels, the time-series case

    probabilities = AttackRMIA._signal_probability(stub, forecasts, true_horizons)

    expected = gaussian_residual_probability(forecasts, true_horizons, 0.5)
    assert probabilities.shape == (6,)
    assert np.allclose(probabilities, expected)
    assert np.all(probabilities > 0.0)
    assert np.all(probabilities <= 1.0)


def test_signal_probability_should_use_laplace_residual_when_criterion_is_l1() -> None:
    stub = SimpleNamespace(is_regression=True, likelihood_family="laplace", temperature=2.0, _scale=0.7, _huber_delta=None)
    rng = np.random.default_rng(2)
    forecasts = rng.normal(size=(5, 4))
    true_horizons = rng.normal(size=(5, 4))

    probabilities = AttackRMIA._signal_probability(stub, forecasts, true_horizons)

    expected = laplace_residual_probability(forecasts, true_horizons, 0.7)
    assert np.allclose(probabilities, expected)


def test_signal_probability_should_use_huber_residual_when_criterion_is_huber() -> None:
    stub = SimpleNamespace(is_regression=True, likelihood_family="huber", temperature=2.0, _scale=0.9, _huber_delta=1.5)
    rng = np.random.default_rng(3)
    forecasts = rng.normal(size=(5, 4))
    true_horizons = rng.normal(size=(5, 4))

    probabilities = AttackRMIA._signal_probability(stub, forecasts, true_horizons)

    expected = huber_residual_probability(forecasts, true_horizons, 0.9, 1.5)
    assert np.allclose(probabilities, expected)


def test_signal_probability_should_raise_when_scale_is_unresolved() -> None:
    stub = SimpleNamespace(is_regression=True, likelihood_family="gaussian", temperature=2.0, _scale=None, _huber_delta=None)

    with pytest.raises(RuntimeError, match="prepare_attack"):
        AttackRMIA._signal_probability(stub, np.zeros((2, 3)), np.zeros((2, 3)))


@pytest.mark.parametrize(
    ("family", "huber_delta", "expected_scale"),
    [
        ("gaussian", None, 9.0),              # sigma^2
        ("laplace", None, 3.0 / np.sqrt(2)),  # b = sigma / sqrt(2), since Var(Laplace) = 2 b^2
        ("huber", 1.0, 9.0),                  # variance of the quadratic core
    ],
    ids=["gaussian", "laplace", "huber"],
)
def test_resolve_scale_should_map_user_sigma_to_family_scale(family, huber_delta, expected_scale) -> None:
    stub = SimpleNamespace(sigma=3.0, epsilon=1e-6, likelihood_family=family, _huber_delta=huber_delta, _scale=None)

    AttackRMIA._resolve_scale(stub, [np.zeros((5, 2))], np.ones((5, 2)))

    assert np.isclose(stub._scale, expected_scale)


def test_resolve_scale_should_estimate_mean_squared_residual_when_gaussian() -> None:
    stub = SimpleNamespace(sigma=None, epsilon=1e-6, likelihood_family="gaussian", _huber_delta=None, _scale=None)
    z_labels = np.zeros((4, 3))
    shadow_one = np.full((4, 3), 1.0)   # squared residual 1 everywhere
    shadow_two = np.full((4, 3), 3.0)   # squared residual 9 everywhere

    AttackRMIA._resolve_scale(stub, [shadow_one, shadow_two], z_labels)

    assert np.isclose(stub._scale, 5.0)  # mean of (1, 9)


def test_resolve_scale_should_estimate_mean_absolute_residual_when_laplace() -> None:
    stub = SimpleNamespace(sigma=None, epsilon=1e-6, likelihood_family="laplace", _huber_delta=None, _scale=None)
    z_labels = np.zeros((4, 3))
    shadow_one = np.full((4, 3), 1.0)   # absolute residual 1 everywhere
    shadow_two = np.full((4, 3), 3.0)   # absolute residual 3 everywhere

    AttackRMIA._resolve_scale(stub, [shadow_one, shadow_two], z_labels)

    assert np.isclose(stub._scale, 2.0)  # mean of (1, 3), the Laplace scale MLE


def test_resolve_scale_should_estimate_mean_huber_energy_when_huber() -> None:
    stub = SimpleNamespace(sigma=None, epsilon=1e-6, likelihood_family="huber", _huber_delta=1.0, _scale=None)
    z_labels = np.zeros((4, 3))
    shadow_one = np.full((4, 3), 1.0)   # |r| = 1 <= delta: energy 0.5 * 1^2 = 0.5
    shadow_two = np.full((4, 3), 3.0)   # |r| = 3 > delta: energy 1 * (3 - 0.5) = 2.5

    AttackRMIA._resolve_scale(stub, [shadow_one, shadow_two], z_labels)

    assert np.isclose(stub._scale, 1.5)  # mean of (0.5, 2.5)


def test_resolve_scale_should_fall_back_to_epsilon_when_shadow_residuals_are_zero() -> None:
    stub = SimpleNamespace(sigma=None, epsilon=1e-6, likelihood_family="gaussian", _huber_delta=None, _scale=None)
    z_labels = np.ones((3, 2))

    AttackRMIA._resolve_scale(stub, [z_labels.copy()], z_labels)

    assert stub._scale == 1e-6
