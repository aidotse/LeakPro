"""Module that contains the AttackScheduler class, which is responsible for creating and executing attacks."""

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.import_helper import Any, Dict, Self
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler


class AttackScheduler:
    """Class responsible for creating and executing attacks."""

    attack_type_to_factory = {"mia": AttackFactoryMIA}

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

        self.logger = handler.logger

        # Create the attacks
        self.attack_list = list(configs["audit"]["attack_list"].keys())
        self.attacks = []
        for attack_name in self.attack_list:
            try:
                attack = factory.create_attack(attack_name, handler)
                self.add_attack(attack)
                self.logger.info(f"Added attack: {attack_name}")
            except ValueError as e:
                self.logger.info(e)
                self.logger.info(f"Failed to create attack: {attack_name}, supported attacks: {factory.attack_classes.keys()}")

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

    def map_setting_to_attacks(self:Self) -> None:
        """Identify relevant attacks based on adversary setting."""
        # TODO: Implement this mapping and remove attack list from configs
        pass
