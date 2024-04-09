"""Module that contains the abstract class for constructing and performing a membership inference attack on a target."""

from abc import ABC, abstractmethod

from leakpro.import_helper import List, Self, Union
from leakpro.metrics.attack_result import AttackResult
from leakpro.mia_attacks.attack_utils import AttackUtils

########################################################################################################################
# METRIC CLASS
########################################################################################################################


class AttackAbstract(ABC):
    """Interface to construct and perform a membership inference attack on a target model and dataset.

    This serves as a guideline for implementing a metric to be used for measuring the privacy leakage of a target model.
    """

    def __init__(
        self:Self,
        attack_utils: AttackUtils,
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            attack_utils (AttackUtils): An instance of the AttackUtils class containing the attack objects.

        """
        self.population = attack_utils.attack_objects.population
        self.population_size = attack_utils.attack_objects.population_size
        self.target_model = attack_utils.attack_objects.target_model
        self.audit_dataset = attack_utils.attack_objects.audit_dataset
        self.signal_data = []


    @property
    def get_population(self:Self)-> List:
        """Get the population used for the attack.

        Returns
        -------
        List: The population used for the attack.

        """
        return self.population

    @property
    def get_population_size(self:Self)-> int:
        """Get the size of the population used for the attack.

        Returns
        -------
        int: The size of the population used for the attack.

        """
        return self.population_size

    @property
    def get_target_model(self:Self)-> Union[Self, List[Self] ]:
        """Get the target model used for the attack.

        Returns
        -------
        Union[Self, List[Self]]: The target model used for the attack.

        """
        return self.target_model

    @property
    def get_audit_dataset(self:Self)-> Self:
        """Get the audit dataset used for the attack.

        Returns
        -------
        Self: The audit dataset used for the attack.

        """
        return self.audit_dataset

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
