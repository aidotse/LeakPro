#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from dotmap import DotMap
from pydantic import ValidationError
from scipy.stats import norm

from leakpro.attacks.mia_attacks.lira import AttackLiRA
from leakpro.reporting.mia_result import MIAResult
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.tests.constants import get_audit_config, get_shadow_model_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler


def test_lira_setup(image_handler:ImageInputHandler) -> None:
    """Test the initialization of LiRA."""
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_obj = AttackLiRA(image_handler, lira_params)

    assert lira_obj is not None
    assert lira_obj.target_model is not None
    assert lira_obj.online == lira_params.online
    assert lira_obj.num_shadow_models == lira_params.num_shadow_models
    assert lira_obj.training_data_fraction == lira_params.training_data_fraction

    description = lira_obj.description()
    assert len(description) == 4

def test_lira_prepare_online_attack(image_handler:ImageInputHandler) -> None:
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_params.online = True

    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    lira_obj.prepare_attack()

    # ensure correct number of shadow models are created
    assert len(lira_obj.shadow_model_indices) == lira_params.num_shadow_models
    # ensure the attack data indices correspond to the correct pool
    assert sorted(lira_obj.attack_data_indices) == list(range(image_handler.population_size))

    # Check that the filtering of the attack data is correct (this is done after shadow models are created)
    n_attack_points = len(lira_obj.train_indices) + len(lira_obj.test_indices)
    assert n_attack_points > 0
    assert lira_obj.shadow_models_signals.shape == (lira_params.num_shadow_models, n_attack_points)
    assert lira_obj.target_signals.shape == (n_attack_points, )

def test_lira_prepare_offline_attack(image_handler:ImageInputHandler) -> None:
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_params.online = False

    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    lira_obj.prepare_attack()

    # ensure correct number of shadow models are created
    assert len(lira_obj.shadow_model_indices) == lira_params.num_shadow_models
    # ensure the attack data indices correspond to the correct pool (all of the data)
    assert sorted(lira_obj.attack_data_indices) == list(range(image_handler.population_size))

    # Check that the filtering of the attack data is correct (this is done after shadow models are created)
    n_attack_points = len(lira_obj.train_indices) + len(lira_obj.test_indices)
    assert n_attack_points > 0
    assert lira_obj.shadow_models_signals.shape == (lira_params.num_shadow_models, n_attack_points)
    assert lira_obj.target_signals.shape == (n_attack_points, )


def test_lira_online_attack(image_handler:ImageInputHandler):
    # Set up for testing
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_params.online = True
    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    lira_obj.prepare_attack()

    # Test standard deviation calculation
    std_fixed = lira_obj.get_std(lira_obj.shadow_models_signals.flatten(),
                           ~lira_obj.out_indices.flatten(),
                           True,
                           "fixed")

    lira_obj.fixed_in_std = std_fixed
    lira_obj.fixed_out_std = std_fixed

    std_carlini = lira_obj.get_std(lira_obj.shadow_models_signals.flatten(),
                           ~lira_obj.out_indices.flatten(),
                           True,
                           "carlini")

    std_individual = lira_obj.get_std(lira_obj.shadow_models_signals.flatten(),
                           ~lira_obj.out_indices.flatten(),
                           True,
                           "individual_carlini")
    assert std_fixed == std_carlini
    assert std_fixed == std_individual

    # Test attack
    lira_obj.run_attack()
    assert lira_obj.fixed_in_std != lira_obj.fixed_out_std
    n_attack_points = len(lira_obj.train_indices) + len(lira_obj.test_indices)
    assert len(lira_obj.in_member_signals)+len(lira_obj.out_member_signals) == n_attack_points
    assert not np.any(np.isnan(lira_obj.in_member_signals))
    assert not np.any(np.isnan(lira_obj.out_member_signals))

def test_lira_offline_attack(image_handler:ImageInputHandler):
    # Set up for testing
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_params.online = False
    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    lira_obj.prepare_attack()
    lira_obj.fix_var_threshold = 0.0

    # Test attack
    lira_result = lira_obj.run_attack()
    assert lira_obj.fixed_in_std != lira_obj.fixed_out_std
    n_attack_points = len(lira_obj.train_indices) + len(lira_obj.test_indices)
    assert len(lira_obj.in_member_signals)+len(lira_obj.out_member_signals) == n_attack_points
    assert not np.any(np.isnan(lira_obj.in_member_signals))
    assert not np.any(np.isnan(lira_obj.out_member_signals))

    assert lira_result is not None
    assert isinstance(lira_result, MIAResult)


def test_lira_rejects_unknown_config_key(image_handler:ImageInputHandler) -> None:
    """An unknown config key must be reported, not silently ignored.

    audit.yaml carried 'individual_mia', which is not a LiRA option and had no effect. With
    extra="forbid" such a key fails at construction instead of quietly doing nothing.
    """
    audit_config = get_audit_config()
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_params.individual_mia = False

    with pytest.raises(ValidationError, match="individual_mia"):
        AttackLiRA(image_handler, lira_params)


def test_lira_rejects_non_scalar_signal(image_handler:ImageInputHandler) -> None:
    """A signal returning a vector per point must fail in prepare_attack, not score the wrong axis.

    'logits' passes the raw per-class logits through, so it yields (n_points, n_classes). Without
    the check, the class axis silently becomes the audit-sample axis in run_attack.
    """
    audit_config = get_audit_config()
    # Strip the "attack" routing key, as AttackScheduler does before constructing the attack
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_params.signal = "logits"
    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    with pytest.raises(ValueError, match="one scalar per audit point"):
        lira_obj.prepare_attack()

def test_lira_offline_score_matches_reference_implementation(image_handler:ImageInputHandler) -> None:
    """Offline LiRA must score -logpdf under the OUT distribution, per tensorflow/privacy mi_lira_2021.

    The reference implementation predicts logpdf(signal; mean_out, std_out) and negates
    at ROC time; LeakPro folds the negation into the score so that higher means member,
    consistent with the online attack. It must NOT use the logcdf (issue #378).
    """
    audit_config = get_audit_config()
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_params.online = False
    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    lira_obj.prepare_attack()
    lira_obj.run_attack()

    n_audit_samples = lira_obj.shadow_models_signals.shape[1]
    actual = np.zeros(n_audit_samples)
    actual[lira_obj.audit_dataset["in_members"]] = lira_obj.in_member_signals.flatten()
    actual[lira_obj.audit_dataset["out_members"]] = lira_obj.out_member_signals.flatten()

    for i in range(n_audit_samples):
        out_mask = lira_obj.out_indices[:, i]
        sm_signals = lira_obj.shadow_models_signals[:, i]
        out_mean = np.mean(sm_signals[out_mask])
        out_std = lira_obj.get_std(sm_signals, out_mask, False, lira_obj.var_calculation)
        expected = -norm.logpdf(lira_obj.target_signals[i], out_mean, out_std + 1e-30)
        assert np.isclose(actual[i], expected), f"sample {i}: got {actual[i]}, reference gives {expected}"


def test_lira_drops_audit_points_without_out_models(image_handler:ImageInputHandler) -> None:
    """Offline LiRA must drop points with no OUT model instead of scoring them as NaN."""
    audit_config = get_audit_config()
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_params.online = False
    lira_obj = AttackLiRA(image_handler, lira_params)

    # 3 shadow models, 4 audit points. Audit point 1 is in every shadow model, so it has no OUT
    # model and its OUT Gaussian would be fitted on an empty slice.
    lira_obj.num_shadow_models = 3
    lira_obj.out_indices = np.array([
        [True, False, True, False],
        [False, False, True, True],
        [True, False, False, True],
    ])
    lira_obj.target_signals = np.array([10.0, 11.0, 12.0, 13.0])
    lira_obj.shadow_models_signals = np.arange(12, dtype=float).reshape(3, 4)
    lira_obj.audit_dataset = {
        "data": np.arange(4),
        "in_members": np.array([0, 1]),
        "out_members": np.array([2, 3]),
    }

    lira_obj._drop_unscorable_audit_points()

    assert lira_obj.out_indices.shape == (3, 3)
    assert list(lira_obj.target_signals) == [10.0, 12.0, 13.0]
    assert lira_obj.shadow_models_signals.shape == (3, 3)
    # One of the two IN members was dropped; both OUT members survive, renumbered after it.
    assert list(lira_obj.in_members) == [0]
    assert list(lira_obj.out_members) == [1, 2]


def test_lira_keeps_every_point_when_all_are_scorable(image_handler:ImageInputHandler) -> None:
    """With both sides present the audit set and its member indices must be left untouched."""
    audit_config = get_audit_config()
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_params.online = True
    lira_obj = AttackLiRA(image_handler, lira_params)

    lira_obj.num_shadow_models = 2
    lira_obj.out_indices = np.array([[True, False], [False, True]])
    lira_obj.target_signals = np.array([1.0, 2.0])
    lira_obj.shadow_models_signals = np.arange(4, dtype=float).reshape(2, 2)
    lira_obj.audit_dataset = {
        "data": np.arange(2),
        "in_members": np.array([0]),
        "out_members": np.array([1]),
    }

    lira_obj._drop_unscorable_audit_points()

    assert lira_obj.out_indices.shape == (2, 2)
    assert list(lira_obj.target_signals) == [1.0, 2.0]
    assert list(lira_obj.in_members) == [0]
    assert list(lira_obj.out_members) == [1]


def test_lira_rejects_training_data_fraction_of_one(image_handler:ImageInputHandler) -> None:
    """A fraction of 1 leaves no OUT reference models, so the config must be rejected up front."""
    audit_config = get_audit_config()
    lira_params = DotMap({k: v for k, v in audit_config.attack_list[0].items() if k != "attack"})
    lira_params.training_data_fraction = 1.0

    with pytest.raises(ValidationError):
        AttackLiRA(image_handler, lira_params)
