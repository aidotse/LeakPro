"""Signal class, which is an abstract class representing any type of signal that can be obtained."""

from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.dataset import Dataset
from leakpro.import_helper import List, Self, Tuple
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


########################################################################################################################
# MODEL_LOGIT CLASS
########################################################################################################################
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
        results = []
        for model in models:
            # Initialize a list to store the logits for the current model
            model_logits = []

            # Iterate over the DataLoader (ensures we use transforms etc)
            # NOTE: Shuffle must be false to maintain indices order
            data_loader = handler.get_dataloader(indices, batch_size=64, shuffle=False)
            for data, _ in data_loader:
                # Get logits for each data point
                logits = model.get_logits(data)
                model_logits.extend(logits)
            model_logits = np.array(model_logits)
            # Append the logits for the current model to the results
            results.append(model_logits)

        return results

########################################################################################################################
# MODEL_NEGATIVERESCALEDLOGIT CLASS
########################################################################################################################
class ModelNegativeRescaledLogits(Signal):
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
        data_loader = handler.get_dataloader(indices, shuffle=False)

        # Iterate over the dataset using the DataLoader (ensures we use transforms etc)
        for data, labels in data_loader:

            # Initialize a list to store the logits for the current model
            model_logits = []
            for model in tqdm(models):

                # Get neg. rescaled logits for each data point
                logits = -model.get_rescaled_logits(data, labels)

                # Append the logits for the current model to the results
                model_logits.append(logits)

            model_logits = np.array(model_logits)
        return model_logits

########################################################################################################################
# MODEL_RESCALEDLOGIT CLASS
########################################################################################################################

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
        data_loader = handler.get_dataloader(indices, shuffle=False)

        # Iterate over the dataset using the DataLoader (ensures we use transforms etc)
        for data, labels in data_loader:

            # Initialize a list to store the logits for the current model
            model_logits = []
            for model in tqdm(models):

                # Get rescaled logits for each data point
                logits = model.get_rescaled_logits(data, labels)

                # Append the logits for the current model to the results
                model_logits.append(logits)

            model_logits = np.array(model_logits)
        return model_logits


########################################################################################################################
# MODEL_LOSS CLASS
########################################################################################################################


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
        results = []
        # Compute the signal for each model
        data_loader = handler.get_dataloader(indices, shuffle=False)

        for model in models:
            for data, labels in data_loader:
                results.append(model.get_loss(data, labels))

        return results
