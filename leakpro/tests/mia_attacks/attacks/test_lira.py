
from pytest import raises
from math import isnan


from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.mia_attacks.lira import AttackLiRA
from leakpro.tests.input_handler.cifar10_input_handler import Cifar10InputHandler
from leakpro.tests.constants import get_shadow_model_config, get_audit_config

def test_lira_setup(image_handler:Cifar10InputHandler) -> None:
    """Test the initialization of LiRA."""
    audit_config = get_audit_config()
    lira_params = audit_config.attack_list.lira
    lira_obj = AttackLiRA(image_handler, lira_params)
    
    assert lira_obj is not None
    assert lira_obj.target_model is not None
    assert lira_obj.online == lira_params.online
    assert lira_obj.num_shadow_models == lira_params.num_shadow_models
    assert lira_obj.training_data_fraction == lira_params.training_data_fraction
    assert lira_obj.memorization == False
    
    lira_params.num_shadow_models = -1
    with raises(ValueError) as excinfo:
        lira_obj = AttackLiRA(image_handler, lira_params)
    assert str(excinfo.value) == "num_shadow_models must be between 1 and None"
    
    lira_params.num_shadow_models = 3
    
    description = lira_obj.description()
    assert len(description) == 4

def test_lira_prepare_online_attack(image_handler:Cifar10InputHandler) -> None:
    audit_config = get_audit_config()
    lira_params = audit_config.attack_list.lira
    lira_params.online = True
    
    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)
    
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
        
    lira_obj.prepare_attack()
    
    # ensure correct number of shadow models are read
    assert len(lira_obj.shadow_models) == lira_params.num_shadow_models
    # ensure the attack data indices correspond to the correct pool
    assert sorted(lira_obj.attack_data_indices) == list(range(image_handler.population_size))
    # memorization is tested in a different module
    assert lira_obj.memorization == False
    
    # Check that the filtering of the attack data is correct (this is done after shadow models are created)
    n_attack_points = len(lira_obj.in_members) + len(lira_obj.out_members)
    assert n_attack_points > 0
    assert lira_obj.shadow_models_logits.shape == (n_attack_points, lira_params.num_shadow_models)
    assert lira_obj.target_logits.shape == (n_attack_points, )
    
def test_lira_prepare_offline_attack(image_handler:Cifar10InputHandler) -> None:
    audit_config = get_audit_config()
    lira_params = audit_config.attack_list.lira
    lira_params.online = False
    
    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)
    
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
        
    lira_obj.prepare_attack()
    
    # ensure correct number of shadow models are read
    assert len(lira_obj.shadow_models) == lira_params.num_shadow_models
    # ensure the attack data indices correspond to the correct pool
    assert sorted(lira_obj.attack_data_indices) == sorted(set(range(image_handler.population_size)) - set(image_handler.test_indices) - set(image_handler.train_indices))
    # memorization is tested in a different module
    assert lira_obj.memorization == False
    
    # Check that the filtering of the attack data is correct (this is done after shadow models are created)
    n_attack_points = len(lira_obj.in_members) + len(lira_obj.out_members)
    assert n_attack_points > 0
    assert lira_obj.shadow_models_logits.shape == (n_attack_points, lira_params.num_shadow_models)
    assert lira_obj.target_logits.shape == (n_attack_points, )


def test_lira_online_attack(image_handler:Cifar10InputHandler):
    # Set up for testing
    audit_config = get_audit_config()
    lira_params = audit_config.attack_list.lira
    lira_params.online = True
    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    lira_obj.prepare_attack()
    
    # Test standard deviation calculation
    std_fixed = lira_obj.get_std(lira_obj.shadow_models_logits.flatten(),
                           lira_obj.in_indices_masks.flatten(),
                           True,
                           "fixed")
    lira_obj.fixed_in_std = std_fixed
    lira_obj.fixed_out_std = std_fixed

    std_carlini = lira_obj.get_std(lira_obj.shadow_models_logits.flatten(),
                           lira_obj.in_indices_masks.flatten(),
                           True,
                           "carlini")
    
    std_individual = lira_obj.get_std(lira_obj.shadow_models_logits.flatten(),
                           lira_obj.in_indices_masks.flatten(),
                           True,
                           "individual_carlini")
    assert std_fixed == std_carlini
    assert std_fixed == std_individual
    
    # Test attack
    lira_obj.run_attack()
    assert lira_obj.fixed_in_std != lira_obj.fixed_out_std
    n_attack_points = len(lira_obj.in_members) + len(lira_obj.out_members)
    assert len(lira_obj.in_member_signals)+len(lira_obj.out_member_signals) == n_attack_points
    assert any(isnan(x) for x in lira_obj.in_member_signals) == False
    assert any(isnan(x) for x in lira_obj.out_member_signals) == False
    
def test_lira_online_attack(image_handler:Cifar10InputHandler):
    # Set up for testing
    audit_config = get_audit_config()
    lira_params = audit_config.attack_list.lira
    lira_params.online = False
    image_handler.configs.shadow_model = get_shadow_model_config()
    lira_obj = AttackLiRA(image_handler, lira_params)
    if ShadowModelHandler.is_created() == False:
        ShadowModelHandler(image_handler)
    lira_obj.prepare_attack()
    lira_obj.fix_var_threshold = 0.0
    
    # Test attack
    lira_obj.run_attack()
    assert lira_obj.fixed_in_std != lira_obj.fixed_out_std
    n_attack_points = len(lira_obj.in_members) + len(lira_obj.out_members)
    assert len(lira_obj.in_member_signals)+len(lira_obj.out_member_signals) == n_attack_points
    assert any(isnan(x) for x in lira_obj.in_member_signals) == False
    assert any(isnan(x) for x in lira_obj.out_member_signals) == False