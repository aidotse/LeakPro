"""Module that contains the AttackFactory class which is responsible for creating the attack objects."""
from logging import Logger

import numpy as np
from torch import nn

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_p import AttackP
from leakpro.attacks.mia_attacks.qmia import AttackQMIA
from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.attacks.utils.shadow_model import ShadowModelHandler
from leakpro.model import PytorchModel


class AttackFactoryMIA:
    """Class responsible for creating the attack objects."""

    attack_classes = {
        "population": AttackP,
        "rmia": AttackRMIA,
        "qmia": AttackQMIA,
    }

    # Shared variables for all attacks
    population = None
    audit_dataset = None
    target_model = None
    target_metadata = None
    logger = None
    shadow_model_handler = None

    @staticmethod
    def set_population_and_audit_data(population:np.ndarray, target_metadata:dict) -> None:
        """Initialize the population dataset."""
        if AttackFactoryMIA.population is None:
            AttackFactoryMIA.population = population

        if AttackFactoryMIA.target_metadata is None:
            AttackFactoryMIA.target_metadata = target_metadata

        if AttackFactoryMIA.audit_dataset is None:
           AttackFactoryMIA.audit_dataset = {
            # Assuming train_indices and test_indices are arrays of indices, not the actual data
            "data": np.concatenate(
                (
                    target_metadata["train_indices"],
                    target_metadata["test_indices"],
                )
            ),
            # in_members will be an array from 0 to the number of training indices - 1
            "in_members": np.arange(len(target_metadata["train_indices"])),
            # out_members will start after the last training index and go up to the number of test indices - 1
            "out_members": np.arange(
                len(target_metadata["train_indices"]),
                len(target_metadata["train_indices"])
                + len(target_metadata["test_indices"]),
            ),
        }

    @staticmethod
    def set_target_model_and_loss(target_model:nn.Module, criterion:nn.Module) -> None:
        """Set the target model."""
        if AttackFactoryMIA.target_model is None:
            AttackFactoryMIA.target_model = PytorchModel(target_model, criterion)

    @staticmethod
    def set_logger(logger:Logger) -> None:
        """Set the logger for the AttackFactoryMIA class."""
        if AttackFactoryMIA.logger is None:
            AttackFactoryMIA.logger = logger

    @classmethod
    def create_attack(cls, name: str, configs: dict) -> AbstractMIA:  # noqa: ANN102
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
        if AttackFactoryMIA.population is None:
            raise ValueError("Population data has not been set")
        if AttackFactoryMIA.audit_dataset is None:
            raise ValueError("Audit data has not been set")
        if AttackFactoryMIA.target_model is None:
            raise ValueError("Target model has not been set")
        if AttackFactoryMIA.logger is None:
            raise ValueError("Logger has not been set")

        if "shadow_model" in configs and AttackFactoryMIA.shadow_model_handler is None:
            AttackFactoryMIA.logger.info("Creating shadow model handler singleton")
            AttackFactoryMIA.shadow_model_handler = ShadowModelHandler(
                                                        AttackFactoryMIA.target_model,
                                                        AttackFactoryMIA.target_metadata,
                                                        configs["shadow_model"],
                                                        AttackFactoryMIA.logger
                                                    )

        if name in cls.attack_classes:
            return cls.attack_classes[name](
                AttackFactoryMIA.population,
                AttackFactoryMIA.audit_dataset,
                AttackFactoryMIA.target_model,
                AttackFactoryMIA.logger,
                configs["audit"]["attack_list"][name]
            )
        raise ValueError(f"Unknown attack type: {name}")
