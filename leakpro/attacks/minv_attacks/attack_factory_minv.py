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

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
from leakpro.attacks.minv_attacks.plgmi import AttackPLGMI
from leakpro.input_handler.minv_handler import MINVHandler


class AttackFactoryMINV:
    """Class responsible for creating the attack objects."""

    attack_classes = {
        "plgmi": AttackPLGMI,
        }

    # Shared variables for all attacks
    generator_handler = None # TODO: Implement this if needed

    @classmethod
    def create_attack(cls, name: str, attack_config: dict, handler: MINVHandler) -> AbstractMINV:
        """Create the attack object.

        Args:
        ----
            name (str): The name of the attack.
            attack_config (dict): Configuration dictionary for the attack.
            handler (AbstractInputHandler): The input handler object.

        Returns:
        -------
            AttackBase: An instance of the attack object.

        Raises:
        ------
            ValueError: If the attack type is unknown.

        """

        if name in cls.attack_classes:
            return cls.attack_classes[name](handler, attack_config)
        raise ValueError(f"Unknown attack type: {name}")
