"""Module that contains the abstract class for constructing and performing a membership inference attack on a target."""

from abc import ABC, abstractmethod
from logging import Logger

import numpy as np
from torch import nn

from leakpro.import_helper import List, Self, Union
from leakpro.metrics.attack_result import AttackResult

########################################################################################################################
# METRIC CLASS
########################################################################################################################


class AbstractMIA(ABC):
    """Interface to construct and perform a membership inference attack on a target model and dataset.

    This serves as a guideline for implementing a metric to be used for measuring the privacy leakage of a target model.
    """

    def __init__(
        self:Self,
        population: np.ndarray,
        audit_dataset: dict,
        target_model: nn.Module,
        logger:Logger
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            population (np.ndarray): The population used for the attack.
            audit_dataset (dict): The audit dataset used for the attack.
            target_model (nn.Module): The target model used for the attack.
            logger (Logger): The logger used for logging.

        """
        self._population = population
        self._population_size = len(population)
        self._target_model = target_model
        self._audit_dataset = audit_dataset
        self.logger = logger
        self.signal_data = []

    @property
    def population(self:Self)-> List:
        """Get the population used for the attack.

        Returns
        -------
        List: The population used for the attack.

        """
        return self._population

    @property
    def population_size(self:Self)-> int:
        """Get the size of the population used for the attack.

        Returns
        -------
        int: The size of the population used for the attack.

        """
        return self._population_size

    @property
    def target_model(self:Self)-> Union[Self, List[Self] ]:
        """Get the target model used for the attack.

        Returns
        -------
        Union[Self, List[Self]]: The target model used for the attack.

        """
        return self._target_model

    @property
    def audit_dataset(self:Self)-> Self:
        """Get the audit dataset used for the attack.

        Returns
        -------
        Self: The audit dataset used for the attack.

        """
        return self._audit_dataset

    @abstractmethod
    def description(self:Self) -> dict:
        """Return a description of the attack.

        Returns
        -------
        dict: A dictionary containing the reference, summary, and detailed description of the attack.

        """
        pass

    @abstractmethod
    def prepare_attack(self:Self) -> None:
        """Prepare data needed for running the metric on the target model and dataset."""
        pass

    @abstractmethod
    def run_attack(self:Self) -> Union[AttackResult, List[AttackResult]]:
        """Run the metric on the target model and dataset.

        Args:
        ----
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
        -------
            Result(s) of the metric.

        """
        pass
