"""Module that contains the AttackScheduler class, which is responsible for creating and executing attacks."""

from leakpro.attacks.gia_attacks.attack_factory_gia import AttackFactoryGIA
from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.utils.import_helper import Any, Dict, Self
from leakpro.utils.logger import logger


class AttackScheduler:
    """Class responsible for creating and executing attacks."""

    attack_type_to_factory = {"mia": AttackFactoryMIA,
                              "gia": AttackFactoryGIA}

    def __init__(
        self:Self,
        handler: AbstractInputHandler,
    ) -> None:
        """Initialize the AttackScheduler class.

        Args:
        ----
            handler (AbstractInputHandler): The handler object that contains the user inputs.

        """
        configs = handler.configs
        if configs["audit"]["attack_type"] not in list(self.attack_type_to_factory.keys()):
            raise ValueError(
                f"Unknown attack type: {configs['audit']['attack_type']}. "
                f"Supported attack types: {self.attack_type_to_factory.keys()}"
            )

        # Prepare factory
        factory = self.attack_type_to_factory[configs["audit"]["attack_type"]]

        # Create the attacks
        self.attack_list = list(configs["audit"]["attack_list"].keys())
        self.attacks = []
        for attack_name in self.attack_list:
            try:
                attack = factory.create_attack(attack_name, handler)
                self.add_attack(attack)
                logger.info(f"Added attack: {attack_name}")
            except ValueError as e:
                logger.info(e)
                logger.info(f"Failed to create attack: {attack_name}, supported attacks: {factory.attack_classes.keys()}")

    def add_attack(self:Self, attack: AbstractMIA) -> None:
        """Add an attack to the list of attacks."""
        self.attacks.append(attack)

    def run_attacks(self:Self) -> Dict[str, Any]:
        """Run the attacks and return the results."""
        results = {}
        for attack, attack_type in zip(self.attacks, self.attack_list):
            logger.info(f"Preparing attack: {attack_type}")
            attack.prepare_attack()

            logger.info(f"Running attack: {attack_type}")

            result = attack.run_attack()
            results[attack_type] = {"attack_object": attack, "result_object": result}

            logger.info(f"Finished attack: {attack_type}")
        return results

    def map_setting_to_attacks(self:Self) -> None:
        """Identify relevant attacks based on adversary setting."""
        # TODO: Implement this mapping and remove attack list from configs
        pass
