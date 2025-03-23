import numpy as np

from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.reporting.attack_result import MIAResult
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.tests.constants import get_audit_config, get_shadow_model_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler
from leakpro.schemas import OptunaConfig
# Run optuna: offline, online
# mask a parameter
# mask all parameters

def test_rmia_optuna_attack_change_default_objective(image_handler:ImageInputHandler) -> None:
    audit_config = get_audit_config()
    # Note: this does not include gamma and a_offline
    rmia_params = audit_config.attack_list.rmia
    rmia_params.online = False

    # Set up attack object
    image_handler.configs.shadow_model = get_shadow_model_config()
    rmia_obj = AttackRMIA(image_handler, rmia_params)
    rmia_obj.set_effective_optuna_metadata(rmia_params)
    
    # as the optuna optimizable parameters have not been set, there should be 2 parameters to optimize
    assert rmia_obj.optuna_params > 0, "No parameters to optimize"
        
    # set gamma and a_offline
    start_gamma = 9
    start_offline_a = 0.9999
    rmia_obj.configs.gamma = start_gamma
    rmia_obj.configs.offline_a = start_offline_a

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    rmia_obj.prepare_attack()
    
    # run optuna
    optuna_config = OptunaConfig() # Set up Optuna config
    optuna_config.n_trials = 2
    study = rmia_obj.run_with_optuna(optuna_config)
    best_config = rmia_obj.configs.model_copy(update=study.best_params)
    
    assert best_config.gamma != start_gamma, "Gamma was not optimized"
    assert best_config.offline_a != start_offline_a, "a_offline was not optimized"
    
    optuna_config.objective = lambda x: 1
    study = rmia_obj.run_with_optuna(optuna_config)
    best_config_dummy_obj = rmia_obj.configs.model_copy(update=study.best_params)
    assert best_config.gamma != best_config_dummy_obj.gamma, "Gamma was not optimized"
    assert best_config.offline_a != best_config_dummy_obj.offline_a, "a_offline was not optimized"
    
    
def test_rmia_optuna_selective_attack(image_handler:ImageInputHandler) -> None:
    audit_config = get_audit_config()
    # Note: this does not include gamma and a_offline
    rmia_params = audit_config.attack_list.rmia
    rmia_params.online = False
    start_offline_a = 0.9999
    rmia_params.offline_a = start_offline_a

    # Set up attack object
    image_handler.configs.shadow_model = get_shadow_model_config()
    rmia_obj = AttackRMIA(image_handler, rmia_params)
    rmia_obj.set_effective_optuna_metadata(rmia_params)
    
    # as the optuna optimizable parameters have not been set, there should be 2 parameters to optimize
    assert rmia_obj.optuna_params > 0, "No parameters to optimize"
        
    # set gamma and a_offline
    start_gamma = 9
    rmia_obj.configs.gamma = start_gamma

    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)

    rmia_obj.prepare_attack()
    
    # run optuna
    optuna_config = OptunaConfig() # Set up Optuna config
    optuna_config.n_trials = 2
    study = rmia_obj.run_with_optuna()
    best_config = rmia_obj.configs.model_copy(update=study.best_params)
    
    assert best_config.gamma != start_gamma, "Gamma was not optimized"
    assert best_config.offline_a == start_offline_a, "a_offline was not optimized"
    
    
def test_rmia_optuna_no_params(image_handler:ImageInputHandler) -> None:
    audit_config = get_audit_config()
    # Note: this does not include gamma and a_offline
    rmia_params = audit_config.attack_list.rmia
    rmia_params.online = False
    start_offline_a = 0.9999
    start_gamma = 9
    rmia_params.offline_a = start_offline_a
    rmia_params.gamma = start_gamma

    # Set up attack object
    image_handler.configs.shadow_model = get_shadow_model_config()
    rmia_obj = AttackRMIA(image_handler, rmia_params)
    rmia_obj.set_effective_optuna_metadata(rmia_params)
    
    # as the optuna optimizable parameters have not been set, there should be 2 parameters to optimize
    assert rmia_obj.optuna_params == 0, "Too many parameters to optimize"

