
import numpy as np

from leakpro.attacks.mia_attacks.rmia import AttackRMIA
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
            "attack_data_fraction": 0.1,
            "online": False
        }
    rmia_params = None
    rmia_obj = AttackRMIA(image_handler, rmia_params) # create RMIA with default params
    # Ensure default parameters are set
    for key in default_params:
        assert getattr(rmia_obj, key) == default_params[key]
    
    # Try custom params
    audit_config = get_audit_config()
    rmia_params = audit_config.attack_list[1]
    rmia_obj = AttackRMIA(image_handler, rmia_params)

    assert rmia_obj is not None
    assert rmia_obj.target_model is not None
    assert rmia_obj.online == rmia_params.online
    assert rmia_obj.num_shadow_models == rmia_params.num_shadow_models
    assert rmia_obj.training_data_fraction == rmia_params.training_data_fraction
    assert rmia_obj.attack_data_fraction == rmia_params.attack_data_fraction

    description = rmia_obj.description()
    assert len(description) == 4

def test_rmia_prepare_online_attack(image_handler:ImageInputHandler) -> None:
    audit_config = get_audit_config()
    rmia_params = audit_config.attack_list[1]
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
    rmia_params = audit_config.attack_list[1]
    rmia_params.online = False

    image_handler.configs.shadow_model = get_shadow_model_config()
    rmia_obj = AttackRMIA(image_handler, rmia_params)

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    rmia_obj.prepare_attack()

    # ensure correct number of shadow models are read
    assert len(rmia_obj.shadow_models) == rmia_params.num_shadow_models
    # ensure the attack data indices correspond to the correct pool
    assert sorted(rmia_obj.attack_data_indices) == sorted(set(range(image_handler.population_size)) - set(image_handler.test_indices) - set(image_handler.train_indices))

    # Check that the filtering of the attack data is correct (this is done after shadow models are created)
    n_attack_points = len(rmia_obj.attack_data_indices) * rmia_params.attack_data_fraction
    assert len(rmia_obj.ratio_z.squeeze()) == n_attack_points
    assert rmia_obj.ratio_z.all() >= 0.0
    assert rmia_obj.ratio_z.all() <= 1.0


def test_rmia_online_attack(image_handler:ImageInputHandler):
    # Set up for testing
    audit_config = get_audit_config()
    rmia_params = audit_config.attack_list[1]
    rmia_params.online = True
    image_handler.configs.shadow_model = get_shadow_model_config()
    rmia_obj = AttackRMIA(image_handler, rmia_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    rmia_obj.prepare_attack()
    # Attack points (not shadow models) are not sampled from train/test data
    non_train_test_points = set(range(image_handler.population_size)) - set(image_handler.test_indices) - set(image_handler.train_indices)
    n_attack_points = len(non_train_test_points) * rmia_params.attack_data_fraction

    # Test attack
    rmia_obj.gamma = 1.0 # model is not trained so no strong signals
    rmia_obj.run_attack()
    
    assert len(rmia_obj.attack_data_index) == n_attack_points
    
    n_audit_points = len(rmia_obj.audit_dataset["data"]) # Total number of points that can be audited
    assert len(rmia_obj.in_member_signals)+len(rmia_obj.out_member_signals) <= n_audit_points # Not all points will have both in and out models
    assert not np.any(np.isnan(rmia_obj.in_member_signals))
    assert not np.any(np.isnan(rmia_obj.out_member_signals))
    assert not np.any(np.isinf(rmia_obj.in_member_signals))
    assert not np.any(np.isinf(rmia_obj.out_member_signals))



def test_rmia_offline_attack(image_handler:ImageInputHandler):
    # Set up for testing
    audit_config = get_audit_config()
    rmia_params = audit_config.attack_list[1]
    rmia_params.online = False
    image_handler.configs.shadow_model = get_shadow_model_config()
    rmia_obj = AttackRMIA(image_handler, rmia_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    rmia_obj.prepare_attack()

    # Test attack
    rmia_obj.gamma = 1.0 # model is not trained so no strong signals
    rmia_result = rmia_obj.run_attack()
    n_attack_points = len(rmia_obj.audit_dataset["data"])
    assert len(rmia_obj.in_member_signals)+len(rmia_obj.out_member_signals) == n_attack_points
    assert not np.any(np.isnan(rmia_obj.in_member_signals))
    assert not np.any(np.isnan(rmia_obj.out_member_signals))
    assert not np.any(np.isinf(rmia_obj.in_member_signals))
    assert not np.any(np.isinf(rmia_obj.out_member_signals))

    assert rmia_result is not None
    assert isinstance(rmia_result, MIAResult)
    