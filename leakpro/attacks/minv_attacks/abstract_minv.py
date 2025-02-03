"""Module that contains the abstract class for constructing and performing a model inversion attack on a target."""

from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import AttackResult
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.import_helper import List, Self, Union

########################################################################################################################
# METRIC CLASS
########################################################################################################################

class AbstractMINV(ABC):
    """Interface to construct and perform a model inversion attack on a target model and dataset.

    This serves as a guideline for implementing a metric to be used for measuring the privacy leakage of a target model.
    """

    # TODO: Class attributes



    def __init__(
        self:Self,
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """


        # TODO: Class attributes initialized checks

    @abstractmethod
    def _configure_attack(self:Self, configs:dict)->None:
        """Configure the attack.

        Args:
        ----
            configs (dict): The configurations for the attack.

        """
        pass

    def _validate_config(self: Self, name: str, value: float, min_val: float, max_val: float) -> None:
        if not (min_val <= value <= (max_val if max_val is not None else value)):
            raise ValueError(f"{name} must be between {min_val} and {max_val}")

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
    def run_attack(self:Self) -> Union[AttackResult, List[AttackResult]]:
        """Run the metric on the target model and dataset. This method handles all the computations related to the audit dataset.

        Args:
        ----
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
        -------
            Result(s) of the metric.

        """

        # TODO: Think about args and return type
        pass