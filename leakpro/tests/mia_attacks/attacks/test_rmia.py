#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#

from types import SimpleNamespace

import numpy as np
import pytest
from dotmap import DotMap
from torch import nn

from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.attacks.utils.utils import gaussian_residual_probability, softmax_logits
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
# Task resolution and the Gaussian residual (regression/forecasting) signal path
# ---------------------------------------------------------------------------

def test_rmia_should_resolve_classification_task_when_criterion_is_cross_entropy(image_handler:ImageInputHandler) -> None:
    rmia_obj = AttackRMIA(image_handler, None)

    assert isinstance(image_handler.get_criterion(), nn.CrossEntropyLoss)
    assert rmia_obj.is_regression is False


def test_rmia_should_resolve_regression_task_when_criterion_is_mse(image_handler:ImageInputHandler) -> None:
    original_criterion = image_handler._criterion
    image_handler._criterion = nn.MSELoss()
    try:
        rmia_obj = AttackRMIA(image_handler, None)
        assert rmia_obj.is_regression is True
        assert rmia_obj._sigma2 is None  # unresolved until prepare_attack
    finally:
        image_handler._criterion = original_criterion


def test_signal_probability_should_match_softmax_indexing_when_classification() -> None:
    stub = SimpleNamespace(is_regression=False, temperature=2.0, _sigma2=None)
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 3))
    labels = np.array([0, 2, 1, 1])

    probabilities = AttackRMIA._signal_probability(stub, logits, labels)

    expected = softmax_logits(logits, 2.0)[np.arange(4), labels]
    assert np.allclose(probabilities, expected)


def test_signal_probability_should_use_gaussian_residual_when_regression() -> None:
    stub = SimpleNamespace(is_regression=True, temperature=2.0, _sigma2=0.5)
    rng = np.random.default_rng(1)
    forecasts = rng.normal(size=(6, 8))      # (n_points, horizon)
    true_horizons = rng.normal(size=(6, 8))  # float labels, the time-series case

    probabilities = AttackRMIA._signal_probability(stub, forecasts, true_horizons)

    expected = gaussian_residual_probability(forecasts, true_horizons, 0.5)
    assert probabilities.shape == (6,)
    assert np.allclose(probabilities, expected)
    assert np.all(probabilities > 0.0)
    assert np.all(probabilities <= 1.0)


def test_signal_probability_should_raise_when_sigma2_is_unresolved() -> None:
    stub = SimpleNamespace(is_regression=True, temperature=2.0, _sigma2=None)

    with pytest.raises(RuntimeError, match="prepare_attack"):
        AttackRMIA._signal_probability(stub, np.zeros((2, 3)), np.zeros((2, 3)))


def test_resolve_sigma2_should_square_user_sigma_when_provided() -> None:
    stub = SimpleNamespace(sigma=3.0, epsilon=1e-6, _sigma2=None)

    AttackRMIA._resolve_sigma2(stub, [np.zeros((5, 2))], np.ones((5, 2)))

    assert stub._sigma2 == 9.0


def test_resolve_sigma2_should_estimate_mean_squared_residual_when_sigma_is_none() -> None:
    stub = SimpleNamespace(sigma=None, epsilon=1e-6, _sigma2=None)
    z_labels = np.zeros((4, 3))
    shadow_one = np.full((4, 3), 1.0)   # squared residual 1 everywhere
    shadow_two = np.full((4, 3), 3.0)   # squared residual 9 everywhere

    AttackRMIA._resolve_sigma2(stub, [shadow_one, shadow_two], z_labels)

    assert np.isclose(stub._sigma2, 5.0)  # mean of (1, 9)


def test_resolve_sigma2_should_fall_back_to_epsilon_when_shadow_residuals_are_zero() -> None:
    stub = SimpleNamespace(sigma=None, epsilon=1e-6, _sigma2=None)
    z_labels = np.ones((3, 2))

    AttackRMIA._resolve_sigma2(stub, [z_labels.copy()], z_labels)

    assert stub._sigma2 == 1e-6
