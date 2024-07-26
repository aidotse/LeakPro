"""Module that contains the AttackFactory class which is responsible for creating the attack objects."""

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler


class AttackFactoryGIA:
    """Class responsible for creating the attack objects."""

    attack_classes = {
        "inverting_gradients": InvertingGradients,
    }

    @classmethod
    def create_attack(cls, name: str, handler: AbstractInputHandler) -> AbstractGIA:  # noqa: ANN102
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
