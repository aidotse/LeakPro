"""Module that contains the AttackScheduler class, which is responsible for creating and executing attacks."""
import logging

from torch import nn

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.dataset import GeneralDataset
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.import_helper import Any, Dict, Self


class AttackScheduler:
    """Class responsible for creating and executing attacks."""

    attack_type_to_factory = {"mia": AttackFactoryMIA}

    def __init__(
        self:Self,
        handler: AbstractInputHandler,
        logger:logging.Logger
    ) -> None:
        """Initialize the AttackScheduler class.

        Args:
        ----
            population (GeneralDataset): The population dataset.
            target_model (torch.nn.Module): The target model.
            target_model_metadata (Dict[str, Any]): The metadata of the target model.
            configs (Dict[str, Any]): The configurations.
            logger (logging.Logger): The logger object.

        """
        configs = handler.configs
        if configs["audit"]["attack_type"] not in list(self.attack_type_to_factory.keys()):
            raise ValueError(
                f"Unknown attack type: {configs['audit']['attack_type']}. "
                f"Supported attack types: {self.attack_type_to_factory.keys()}"
            )

        # Prepare factory
        factory = self.attack_type_to_factory[configs["audit"]["attack_type"]]
        factory.setup(handler)

        self.logger = logger

        # Create the attacks
        self.attack_list = list(configs["audit"]["attack_list"].keys())
        self.attacks = []
        for attack_name in self.attack_list:
            try:
                attack = factory.create_attack(attack_name, configs)
                self.add_attack(attack)
                self.logger.info(f"Added attack: {attack_name}")
            except ValueError as e:
                logger.info(e)

    def add_attack(self:Self, attack: AbstractMIA) -> None:
        """Add an attack to the list of attacks."""
        self.attacks.append(attack)

    def run_attacks(self:Self) -> Dict[str, Any]:
        """Run the attacks and return the results."""
        results = {}
        for attack, attack_type in zip(self.attacks, self.attack_list):
            self.logger.info(f"Preparing attack: {attack_type}")
            attack.prepare_attack()

            self.logger.info(f"Running attack: {attack_type}")

            result = attack.run_attack()
            results[attack_type] = {"attack_object": attack, "result_object": result}

            self.logger.info(f"Finished attack: {attack_type}")
        return results

    def identify_attacks(self:Self) -> None:
        """Identify relevant attacks based on adversary setting."""
        # TODO: Implement this mapping and remove attack list from configs
        pass
