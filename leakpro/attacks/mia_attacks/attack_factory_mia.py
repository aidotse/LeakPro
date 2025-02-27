"""Module that contains the AttackFactory class which is responsible for creating the attack objects."""

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_p import AttackP
from leakpro.attacks.mia_attacks.HSJ import AttackHopSkipJump
from leakpro.attacks.mia_attacks.lira import AttackLiRA
from leakpro.attacks.mia_attacks.loss_trajectory import AttackLossTrajectory
from leakpro.attacks.mia_attacks.oslo import AttackOSLO
from leakpro.attacks.mia_attacks.qmia import AttackQMIA
from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.attacks.mia_attacks.yoqo import AttackYOQO
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.logger import logger


class AttackFactoryMIA:
    """Class responsible for creating the attack objects."""

    attack_classes = {
        "population": AttackP,
        "rmia": AttackRMIA,
        "qmia": AttackQMIA,
        "loss_traj":AttackLossTrajectory,
        "lira": AttackLiRA,
        "HSJ" : AttackHopSkipJump,
        "yoqo": AttackYOQO,
        "oslo": AttackOSLO
    }

    # Shared variables for all attacks
    shadow_model_handler = None
    distillation_model_handler = None

    @classmethod
    def create_attack(cls, name: str, handler: AbstractInputHandler) -> AbstractMIA:  # noqa: ANN102
        """Create the attack object.

        Args:
        ----
            name (str): The name of the attack.
            handler (AbstractInputHandler): The input handler object.

        Returns:
        -------
            AttackBase: An instance of the attack object.

        Raises:
        ------
            ValueError: If the attack type is unknown.

        """

        if AttackFactoryMIA.shadow_model_handler is None:
            logger.info("Creating shadow model handler singleton")
            AttackFactoryMIA.shadow_model_handler = ShadowModelHandler(handler)

        if AttackFactoryMIA.distillation_model_handler is None:
            logger.info("Creating distillation model handler singleton")
            AttackFactoryMIA.distillation_model_handler = DistillationModelHandler(handler)

        if name in cls.attack_classes:
            return cls.attack_classes[name](handler, handler.configs["audit"]["attack_list"][name])
        raise ValueError(f"Unknown attack type: {name}")
