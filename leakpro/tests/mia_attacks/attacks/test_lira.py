
from dotmap import DotMap

from leakpro.user_inputs.cifar10_input_handler import Cifar10InputHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.mia_attacks.lira import AttackLiRA

lira_params = DotMap()
lira_params.training_data_fraction = 0.1
lira_params.num_shadow_models = 3
lira_params.online = False
lira_params.fixed_variance = True

def test_lira_setup(image_handler:Cifar10InputHandler, shadow_handler:ShadowModelHandler) -> None:
    """Test the initialization of LiRA."""
    
    lira_obj = AttackLiRA(image_handler, lira_params)