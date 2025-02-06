"""Module that contains the AttackFactory class which is responsible for creating the attack objects."""

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
# from leakpro.attacks.minv_attacks.plgmi import AttackPLGMI
# from leakpro.attacks.minv_attacks.ifgmi import AttackIFGMI
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler

class AttackFactoryGIA:
    """Class responsible for creating the attack objects."""

    attack_classes = {
        "plg": AttackPLGMI,
        "ifg": AttackIFGMI,
    }

    @classmethod
    def create_attack(cls, name: str, handler: AbstractInputHandler) -> AbstractMINV:
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

        if name in cls.attack_classes:
            return cls.attack_classes[name](handler, handler.configs["audit"]["attack_list"][name])
        raise ValueError(f"Unknown attack type: {name}")
