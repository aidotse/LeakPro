"""Module that contains the abstract class for constructing and performing a membership inference attack on a target."""

from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader

from leakpro.import_helper import List, Self, Union
from leakpro.metrics.attack_result import AttackResult
from leakpro.model import PytorchModel
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler

########################################################################################################################
# METRIC CLASS
########################################################################################################################


class AbstractMIA(ABC):
    """Interface to construct and perform a membership inference attack on a target model and dataset.

    This serves as a guideline for implementing a metric to be used for measuring the privacy leakage of a target model.
    """

    # Class attributes for sharing between the different attacks
    population = None
    population_size = None
    target_model = None
    audit_dataset = None
    _initialized = False

    def __init__(
        self:Self,
        handler: AbstractInputHandler,
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """
        # These objects are shared and should be initialized only once
        if not AbstractMIA._initialized:
            AbstractMIA.population = handler.population
            AbstractMIA.population_size = handler.population_size
            AbstractMIA.target_model = PytorchModel(handler.target_model, handler.criterion)
            AbstractMIA.audit_dataset = {
                # Assuming train_indices and test_indices are arrays of indices, not the actual data
                "data": np.concatenate((handler.train_indices, handler.test_indices)),
                # in_members will be an array from 0 to the number of training indices - 1
                "in_members": np.arange(len(handler.train_indices)),
                # out_members will start after the last training index and go up to the number of test indices - 1
                "out_members": np.arange(len(handler.train_indices),len(handler.train_indices)+len(handler.test_indices)),
            }
            self.handler = handler
            self._validate_shared_quantities()
            AbstractMIA._initialized = True

        # These objects are instance specific
        self.logger = handler.logger
        self.signal_data = []

    def _validate_shared_quantities(self:Self)->None:
        """Validate the shared quantities used by the attack."""
        if AbstractMIA.population is None:
            raise ValueError("Population dataset not found.")
        if AbstractMIA.population_size is None:
            raise ValueError("Population size not found.")
        if AbstractMIA.population_size != len(AbstractMIA.population):
            raise ValueError("Population size does not match the population dataset.")
        if len(AbstractMIA.audit_dataset["in_members"]) == 0:
            raise ValueError("Train indices must be provided.")
        if len(AbstractMIA.audit_dataset["out_members"]) == 0:
            raise ValueError("Test indices must be provided.")
        if AbstractMIA.target_model is None:
            raise ValueError("Target model not found.")
        if AbstractMIA.audit_dataset is None:
            raise ValueError("Audit dataset not found.")

    def sample_indices_from_population(self:Self, *, include_train_indices: bool = False, include_test_indices: bool = False) -> np.ndarray:
        """Function to get attack data for the attack models.

        Args:
        ----
            include_train_indices (bool): Flag indicating whether to include train data in data.
            include_test_indices (bool): Flag indicating whether to include test data in data.

        Returns:
        -------
            np.ndarray: The selected attack data indices.

        """
        all_index = np.arange(AbstractMIA.population_size)

        not_allowed_indices = np.array([])
        if not include_train_indices:
            not_allowed_indices = np.hstack([not_allowed_indices, self.handler.train_indices])

        if not include_test_indices:
            not_allowed_indices = np.hstack([not_allowed_indices, self.handler.test_indices])

        available_index = np.setdiff1d(all_index, not_allowed_indices)
        data_size = len(available_index)
        return np.random.choice(available_index, data_size, replace=False)

    def sample_data_from_dataset(self:Self, data:np.ndarray, size:int)->DataLoader:
        """Function to sample from the dataset.

        Args:
        ----
            data (np.ndarray): The dataset indices to sample from.
            size (int): The size of the sample.

        Returns:
        -------
            Dataloader: The sampled data.

        """
        if size > len(data):
            raise ValueError("Size of the sample is greater than the size of the data.")
        return self.handler.get_dataloader(np.random.choice(data, size, replace=False))

    def get_dataloader(self:Self, data:np.ndarray)->DataLoader:
        """Function to get a dataloader from the dataset.

        Args:
        ----
            data (np.ndarray): The dataset indices to sample from.

        Returns:
        -------
            Dataloader: The sampled data.

        """
        return self.handler.get_dataloader(data)

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

    @property
    def train_indices(self:Self)-> np.ndarray:
        """Get the training indices of the audit dataset.

        Returns
        -------
        np.ndarray: The training indices of the audit dataset.

        """
        train_indices = self._audit_dataset["in_members"]
        return self._audit_dataset["data"][train_indices]


    @property
    def test_indices(self:Self)-> np.ndarray:
        """Get the test indices of the audit dataset.

        Returns
        -------
        np.ndarray: The test indices of the audit dataset.

        """
        test_indices = self._audit_dataset["out_members"]
        return self._audit_dataset["data"][test_indices]

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
        pass
