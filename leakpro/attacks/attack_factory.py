"""Module that contains the AttackFactory class which is responsible for creating the attack objects."""

from leakpro.mia_attacks.attack_utils import AttackUtils
from leakpro.mia_attacks.attacks.attack import AbstractMIA
from leakpro.mia_attacks.attacks.attack_p import AttackP
from leakpro.mia_attacks.attacks.qmia import AttackQMIA
from leakpro.mia_attacks.attacks.rmia import AttackRMIA


class AttackFactory:
    """Class responsible for creating the attack objects."""

    attack_classes = {
        "attack_p": AttackP,
        "rmia": AttackRMIA,
        "qmia": AttackQMIA,
    }

    @classmethod
    def create_attack(cls, name: str, attack_utils: AttackUtils, configs: dict) -> AbstractMIA:  # noqa: ANN102
        """Create an attack object based on the given name, attack_utils, and configs.

        Args:
        ----
            name (str): The name of the attack.
            attack_utils (AttackUtils): An instance of AttackUtils.
            configs (dict): The attack configurations.

        Returns:
        -------
            AttackBase: An instance of the attack object.

        Raises:
        ------
            ValueError: If the attack type is unknown.

        """
        if name in cls.attack_classes:
            return cls.attack_classes[name](attack_utils, configs)
        raise ValueError(f"Unknown attack type: {name}")
