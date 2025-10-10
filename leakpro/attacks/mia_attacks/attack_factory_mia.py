"""Module that contains the AttackFactory class which is responsible for creating the attack objects."""


from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_p import AttackP
from leakpro.attacks.mia_attacks.base import AttackBASE
from leakpro.attacks.mia_attacks.camia import AttackCAMIA
from leakpro.attacks.mia_attacks.HSJ import AttackHopSkipJump
from leakpro.attacks.mia_attacks.lira import AttackLiRA
from leakpro.attacks.mia_attacks.loss_trajectory import AttackLossTrajectory
from leakpro.attacks.mia_attacks.qmia import AttackQMIA
from leakpro.attacks.mia_attacks.ramia import AttackRaMIA
from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.attacks.mia_attacks.yoqo import AttackYOQO
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.schemas import MIAAttackCreationConfig
from leakpro.utils.logger import logger


class AttackFactoryMIA:
    """Class responsible for creating the attack objects."""

    attack_classes = {
        "population": MIAAttackCreationConfig(attack_class=AttackP,
                                              requires_shadow_models=False,
                                              requires_distillation_model=False),
        "rmia": MIAAttackCreationConfig(attack_class=AttackRMIA,
                                        requires_shadow_models=True,
                                        requires_distillation_model=False),
        "qmia": MIAAttackCreationConfig(attack_class=AttackQMIA,
                                        requires_shadow_models=False,
                                        requires_distillation_model=False),
        "loss_traj": MIAAttackCreationConfig(attack_class=AttackLossTrajectory,
                                             requires_shadow_models=True,
                                             requires_distillation_model=True),
        "lira": MIAAttackCreationConfig(attack_class=AttackLiRA,
                                        requires_shadow_models=True,
                                        requires_distillation_model=False),
        "HSJ": MIAAttackCreationConfig(attack_class=AttackHopSkipJump,
                                       requires_shadow_models=False,
                                       requires_distillation_model=False),
        "yoqo": MIAAttackCreationConfig(attack_class=AttackYOQO,
                                        requires_shadow_models=False,
                                        requires_distillation_model=True),
        "base": MIAAttackCreationConfig(attack_class=AttackBASE,
                                        requires_shadow_models=True,
                                        requires_distillation_model=False),
        "ramia": MIAAttackCreationConfig(attack_class=AttackRaMIA,
                                         requires_shadow_models=True,
                                         requires_distillation_model=False),
        "camia": MIAAttackCreationConfig(attack_class=AttackCAMIA,
                                         requires_shadow_models=False,
                                         requires_distillation_model=False)
    }


    # Shared variables for all attacks
    shadow_model_handler = None
    distillation_model_handler = None

    @classmethod
    def create_attack(cls, name: str, attack_config: dict, handler: MIAHandler) -> AbstractMIA:  # noqa: ANN102
        """Create the attack object.

        Args:
        ----
            name (str): The name of the attack.
            attack_config (dict): The configuration for the attack.
            handler (MIAHandler): The input handler object.

        Returns:
        -------
            AttackBase: An instance of the attack object.

        Raises:
        ------
            ValueError: If the attack type is unknown.

        """

        if AttackFactoryMIA.shadow_model_handler is None and cls.attack_classes[name].requires_shadow_models:
            logger.info("Creating shadow model handler singleton")
            AttackFactoryMIA.shadow_model_handler = ShadowModelHandler(handler)

        if AttackFactoryMIA.distillation_model_handler is None and cls.attack_classes[name].requires_distillation_model:
            logger.info("Creating distillation model handler singleton")
            AttackFactoryMIA.distillation_model_handler = DistillationModelHandler(handler)

        if name in cls.attack_classes:
            attack_object = cls.attack_classes[name].attack_class(handler, attack_config)
            attack_object.set_effective_optuna_metadata(attack_config) # remove optuna metadata if params not will be optimized
            return attack_object
        raise ValueError(f"Unknown attack type: {name}")
