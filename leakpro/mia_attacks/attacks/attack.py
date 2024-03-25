from abc import ABC, abstractmethod
from typing import List, Union

from ...metrics.attack_result import AttackResult
from ..attack_utils import AttackUtils

########################################################################################################################
# METRIC CLASS
########################################################################################################################


class AttackAbstract(ABC):
    """Interface to construct and perform a membership inference attack on a target model and dataset using auxiliary
    information specified by the user. This serves as a guideline for implementing a metric to be used for measuring
    the privacy leakage of a target model.
    """

    def __init__(
        self,
        attack_utils: AttackUtils,
    ):
        self.population = attack_utils.attack_objects.population
        self.population_size = attack_utils.attack_objects.population_size
        self.target_model = attack_utils.attack_objects.target_model
        self.audit_dataset = attack_utils.attack_objects.audit_dataset
        self.signal_data = []

    @property
    def get_population(self):
        return self.population

    @property
    def get_population_size(self):
        return self.population_size

    @property
    def get_target_model(self):
        return self.target_model

    @property
    def get_audit_dataset(self):
        return self.audit_dataset

    @abstractmethod
    def prepare_attack(self):
        """Function to prepare data needed for running the metric on the target model and dataset, using signals computed
        on the auxiliary model(s) and dataset.
        """
        pass

    @abstractmethod
    def run_attack(
        self, fpr_tolerance_rate_list=None
    ) -> Union[AttackResult, List[AttackResult]]:
        """Function to run the metric on the target model and dataset.

        Args:
        ----
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
        -------
            Result(s) of the metric.

        """
        pass
