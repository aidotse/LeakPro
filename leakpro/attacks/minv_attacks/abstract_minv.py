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

    # Class attributes for sharing between the different attacks
    public_population = None
    public_population_size = None
    target_model = None
    target_dataset = None
    handler=None
    _initialized = False



    def __init__(
        self:Self,
        handler: AbstractInputHandler
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """
        # These objects are shared and should be initialized only once
        if not AbstractMINV._initialized:
            AbstractMINV.public_data = handler.population
            AbstractMINV.public_size = handler.population_size
            AbstractMINV.target_model = PytorchModel(handler.target_model, handler.get_criterion())
            AbstractMINV.target_dataset = handler.audit_dataset
            AbstractMINV.handler = handler
            AbstractMINV._initialized = True

        # TODO: Class attributes initialized checks


    def get_public_dataloader(self:Self, data:np.ndarray, batch_size:int=None)->DataLoader:
        """Function to get a dataloader from the public dataset.

        Args:
        ----
            data (np.ndarray): The dataset indices to sample from.
            batch_size (int): batch size.

        Returns:
        -------
            Dataloader: The sampled data.

        """
        return self.handler.get_public_dataloader(data) if batch_size is None else self.handler.get_public_dataloader(data, batch_size)

    def get_target_dataloader(self:Self, data:np.ndarray, batch_size:int=None)->DataLoader:
        """Function to get a dataloader from the target dataset.

        Args:
        ----
            data (np.ndarray): The dataset indices to sample from.
            batch_size (int): batch size.

        Returns:
        -------
            Dataloader: The sampled data.

        """
        return self.handler.get_target_dataloader(data) if batch_size is None else self.handler.get_target_dataloader(data, batch_size)

    @property
    def public_population(self:Self)-> List:
        """Get the public population used for the attack.

        Returns
        -------
        List: The public population used for the attack.

        """
        return AbstractMINV.public_population

    @property
    def public_population_size(self:Self)-> int:
        """Get the size of the public population used for the attack.

        Returns
        -------
        int: The size of the public population used for the attack.

        """
        return AbstractMINV.public_population_size

    @property
    def target_model(self:Self)-> Union[Self, List[Self] ]:
        """Get the target model used for the attack.

        Returns
        -------
        Union[Self, List[Self]]: The target model used for the attack.

        """
        return AbstractMINV.target_model

    @property
    def target_dataset(self:Self)-> Union[Self, List[Self] ]:
        """Get the target dataset used for the attack.

        Returns
        -------
        Union[Self, List[Self]]: The target dataset used for the attack.

        """
        return AbstractMINV.target_dataset



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