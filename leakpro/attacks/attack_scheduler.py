"""Module that contains the AttackScheduler class, which is responsible for creating and executing attacks."""

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.import_helper import Any, Dict, Self
from leakpro.utils.logger import logger


class AttackScheduler:
    """Class responsible for creating and executing attacks."""

    attack_type_to_factory = {}

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

        # Create factory
        attack_type = configs["audit"]["attack_type"].lower()
        self._initialize_factory(attack_type)

        # Create the attacks
        self.attack_list = list(configs["audit"]["attack_list"].keys())
        self.attacks = []
        for attack_name in self.attack_list:
            try:
                attack = self.attack_factory.create_attack(attack_name, handler)
                self.add_attack(attack)
                logger.info(f"Added attack: {attack_name}")
            except ValueError as e:
                logger.info(e)
                logger.info(f"Failed to create attack: {attack_name}, supported attacks: {self.attack_factory.attack_classes.keys()}")  # noqa: E501

    def _initialize_factory(self:Self, attack_type:str) -> None:
        """Conditionally import attack factories based on attack."""
        if attack_type == "mia":
            try:
                from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
                self.attack_factory = AttackFactoryMIA
                logger.info("MIA attack factory loaded.")
            except ImportError as e:
                logger.error("Failed to import MIA attack module.")
                raise ImportError("MIA attack module is not available.") from e

        elif attack_type == "gia":
            try:
                from leakpro.attacks.gia_attacks.attack_factory_gia import AttackFactoryGIA
                self.attack_factory = AttackFactoryGIA
                logger.info("GIA attack factory loaded.")
            except ImportError as e:
                logger.error("Failed to import GIA attack module.")
                raise ImportError("GIA attack module is not available.") from e

        else:
            logger.error(f"Unsupported attack type: {self.attack_type}")
            raise ValueError(f"Unsupported attack type: {self.attack_type}. Must be 'mia' or 'gia'.")

    def add_attack(self:Self, attack: Any) -> None:
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
