"""Signal class, which is an abstract class representing any type of signal that can be obtained."""

from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from leakpro.import_helper import List, Self
from leakpro.model import Model
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler

########################################################################################################################
# SIGNAL CLASS
########################################################################################################################


class Signal(ABC):
    """Abstract class, representing any type of signal that can be obtained from a Model and/or a Dataset."""

    @abstractmethod
    def __call__(  # noqa: ANN204
        self: Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        extra: dict,
    ):
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            extra: Dictionary containing any additional parameter that should be passed to the signal object.

        Returns:
        -------
            The signal value.

        """
        pass

    def _is_shuffling(self:Self, dataloader:DataLoader)->bool:
        """Check if the DataLoader is shuffling the data."""
        return not isinstance(dataloader.sampler, SequentialSampler)


class ModelLogits(Signal):
    """Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the output of a model.
    """

    def __call__(
        self: Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.

        Returns:
        -------
            The signal value.

        """        # Compute the signal for each model

        # Iterate over the DataLoader (ensures we use transforms etc)
        # NOTE: Shuffle must be false to maintain indices order
        data_loader = handler.get_dataloader(indices)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a list to store the logits sfor the current model
            model_logits = []

            for data, _ in tqdm(data_loader, desc=f"Getting logits for model {m+1}/ {len(models)}"):
                # Get logits for each data point
                logits = model.get_logits(data)
                model_logits.extend(logits)
            model_logits = np.array(model_logits)
            # Append the logits for the current model to the results
            results.append(model_logits)

        return results

class ModelRescaledLogits(Signal):
    """Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the output of a model.
    """

    def __call__(
        self: Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.

        Returns:
        -------
            The signal value.

        """
        data_loader = handler.get_dataloader(indices)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a list to store the logits for the current model
            model_logits = []

            for data, labels in tqdm(data_loader, desc=f"Getting rescaled logits for model {m+1}/ {len(models)}"):
                # Get logits for each data point
                logits = model.get_rescaled_logits(data,labels)
                model_logits.extend(logits)
            model_logits = np.array(model_logits)
            # Append the logits for the current model to the results
            results.append(model_logits)

        return results

class ModelLoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the loss of a model.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.

        Returns:
        -------
            The signal value.

        """
        # Compute the signal for each model
        data_loader = handler.get_dataloader(indices)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a list to store the logits for the current model
            model_logits = []

            for data, labels in tqdm(data_loader, desc=f"Getting loss for model {m+1}/ {len(models)}"):
                # Get logits for each data point
                loss = model.get_loss(data,labels)
                model_logits.extend(loss)
            model_logits = np.array(model_logits)
            # Append the logits for the current model to the results
            results.append(model_logits)

        return results
