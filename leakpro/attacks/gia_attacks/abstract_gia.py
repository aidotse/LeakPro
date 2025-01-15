"""Module that contains the abstract class for constructing and performing a membership inference attack on a target."""

from abc import ABC, abstractmethod

import optuna

from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.import_helper import Self

########################################################################################################################
# METRIC CLASS
########################################################################################################################


class AbstractGIA(ABC):
    """Interface to construct and perform a gradient inversion attack on a target model and dataset.

    This serves as a guideline for implementing a metric to be used for measuring the privacy leakage of a target model.
    """

    def __init__(  # noqa: B027
        self:Self,
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """
        pass

    @abstractmethod
    def _configure_attack(self:Self, configs:dict)->None:
        """Configure the attack.

        Args:
        ----
            configs (dict): The configurations for the attack.

        """
        pass


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
        """Method that handles all computation related to the attack dataset."""
        pass

    @abstractmethod
    def run_attack(self:Self) -> GIAResults:
        """Run the metric on the target model and dataset. This method handles all the computations related to the audit dataset.

        Args:
        ----
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
        -------
            Result(s) of the metric.

        """
        pass

    @abstractmethod
    def reset_attack(self: Self) -> None:
        """Reset attack to its initial state."""
        pass

    @abstractmethod
    def suggest_parameters(self: Self, trial: optuna.trial.Trial) -> None:
        """Apply and suggest new hyperparameters for the attack using optuna trial."""
        pass

    @abstractmethod
    def get_configs(self:Self) -> dict:
        """Get the configs used for the attack."""
        pass
