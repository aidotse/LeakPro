#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Module that contains the AttackFactory class which is responsible for creating the attack objects."""


from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_p import AttackP
from leakpro.attacks.mia_attacks.base import AttackBASE
from leakpro.attacks.mia_attacks.dts import AttackDTS
from leakpro.attacks.mia_attacks.HSJ import AttackHopSkipJump
from leakpro.attacks.mia_attacks.lira import AttackLiRA
from leakpro.attacks.mia_attacks.loss_trajectory import AttackLossTrajectory
from leakpro.attacks.mia_attacks.multi_signal_lira import AttackMSLiRA
from leakpro.attacks.mia_attacks.oslo import AttackOSLO
from leakpro.attacks.mia_attacks.qmia import AttackQMIA
from leakpro.attacks.mia_attacks.ramia import AttackRaMIA
from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.attacks.mia_attacks.seq_mia import AttackSeqMIA
from leakpro.attacks.mia_attacks.yoqo import AttackYOQO
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.utils.logger import logger


class AttackFactoryMIA:
    """Class responsible for creating the attack objects."""

    attack_classes = {
        "population": AttackP,
        "rmia": AttackRMIA,
        "qmia": AttackQMIA,
        "loss_traj":AttackLossTrajectory,
        "seqmia":AttackSeqMIA,
        "lira": AttackLiRA,
        "HSJ" : AttackHopSkipJump,
        "yoqo": AttackYOQO,
        "base": AttackBASE,
        "ramia": AttackRaMIA,
        "multi_signal_lira": AttackMSLiRA,
        "dts": AttackDTS,
        "oslo": AttackOSLO,
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

        if AttackFactoryMIA.shadow_model_handler is None:
            logger.info("Creating shadow model handler singleton")
            AttackFactoryMIA.shadow_model_handler = ShadowModelHandler(handler)
        else:
            logger.info("Shadow model handler singleton already exists, updating state")
            AttackFactoryMIA.shadow_model_handler = ShadowModelHandler(handler)

        if AttackFactoryMIA.distillation_model_handler is None:
            logger.info("Creating distillation model handler singleton")
            AttackFactoryMIA.distillation_model_handler = DistillationModelHandler(handler)
        else:
            logger.info("Distillation model handler singleton already exists, updating state")
            AttackFactoryMIA.distillation_model_handler = DistillationModelHandler(handler)

        if name in cls.attack_classes:
            attack_object = cls.attack_classes[name](handler, attack_config)
            attack_object.set_effective_optuna_metadata(attack_config) # remove optuna metadata if params not will be optimized
            return attack_object
        raise ValueError(f"Unknown attack type: {name}")
