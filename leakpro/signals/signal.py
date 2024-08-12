"""Signal class, which is an abstract class representing any type of signal that can be obtained."""

import logging as logger
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.dataset import Dataset
from leakpro.import_helper import List, Optional, Self, Tuple
from leakpro.signal_extractor import Model

########################################################################################################################
# SIGNAL CLASS
########################################################################################################################


class Signal(ABC):
    """Abstract class, representing any type of signal that can be obtained from a Model and/or a Dataset."""

    @abstractmethod
    def __call__(  # noqa: ANN204
        self: Self,
        models: List[Model],
        datasets: List[Dataset],
        extra: dict,
    ):
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            datasets: List of datasets that can be queried.
            model_to_split_mapping: List of tuples, indicating how each model should query the dataset.
                More specifically, for model #i:
                model_to_split_mapping[i][0] contains the index of the dataset in the list,
                model_to_split_mapping[i][1] contains the name of the split,
                model_to_split_mapping[i][2] contains the name of the input feature,
                model_to_split_mapping[i][3] contains the name of the output feature.
                This can also be provided once and for all at the instantiation of InformationSource, through the
                default_model_to_split_mapping argument.
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
        datasets: Dataset,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            datasets: List of datasets that can be queried.
            extra: Dictionary containing any additional parameter that should be passed to the signal object.

        Returns:
        -------
            The signal value.

        """        # Compute the signal for each model
        results = []
        for model in models:
            # Initialize a list to store the logits for the current model
            model_logits = []

            # Iterate over the dataset using the DataLoader (ensures we use transforms etc)
            data_loader = DataLoader(datasets, batch_size=len(datasets), shuffle=False)
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
        datasets: Dataset,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            datasets: List of datasets that can be queried.

        Returns:
        -------
            The signal value.

        """
        data_loader = DataLoader(datasets, batch_size=len(datasets), shuffle=False)

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
        datasets: Dataset,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            datasets: datasets that can be queried.

        Returns:
        -------
            The signal value.

        """
        data_loader = DataLoader(datasets, batch_size=len(datasets), shuffle=False)

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
# MODEL_INTERMEDIATE_OUTPUT CLASS
########################################################################################################################

class ModelIntermediateOutput(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the value of an intermediate layer of model.
    """

    def __call__(
        self:Self,
        models: List[Model],
        datasets: List[Dataset],
        model_to_split_mapping: List[Tuple[int, str, str, str]],
        extra: dict,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            datasets: List of datasets that can be queried.
            model_to_split_mapping: List of tuples, indicating how each model should query the dataset.
                More specifically, for model #i:
                model_to_split_mapping[i][0] contains the index of the dataset in the list,
                model_to_split_mapping[i][1] contains the name of the split,
                model_to_split_mapping[i][2] contains the name of the input feature,
                model_to_split_mapping[i][3] contains the name of the output feature.
                This can also be provided once and for all at the instantiation of InformationSource, through the
                default_model_to_split_mapping argument.
            extra: Dictionary containing any additional parameter that should be passed to the signal object.

        Returns:
        -------
            The signal value.

        """
        if "layers" not in list(extra):
            raise TypeError('extra parameter "layers" is required')

        results = []
        # Compute the signal for each model
        for k, model in enumerate(models):
            # Extract the features to be used
            (
                dataset_index,
                split_name,
                input_feature,
                output_feature,
            ) = model_to_split_mapping[k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)
            # Compute the signal
            results.append(model.get_intermediate_outputs(extra["layers"], x))
        return results


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
        dataset: Dataset,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            dataset: datasets to be queried.

        Returns:
        -------
            The signal value.

        """
        results = []
        # Compute the signal for each model
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for model in models:
            for data, labels in data_loader:
                results.append(model.get_loss(data, labels))

        return results


########################################################################################################################
# MODEL_GRADIENT CLASS
########################################################################################################################

class ModelGradient(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the gradient of a model.
    """

    def __call__(
        self:Self,
        models: List[Model],
        datasets: List[Dataset],
        model_to_split_mapping: List[Tuple[int, str, str, str]],
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            datasets: List of datasets that can be queried.
            model_to_split_mapping: List of tuples, indicating how each model should query the dataset.
                More specifically, for model #i:
                model_to_split_mapping[i][0] contains the index of the dataset in the list,
                model_to_split_mapping[i][1] contains the name of the split,
                model_to_split_mapping[i][2] contains the name of the input feature,
                model_to_split_mapping[i][3] contains the name of the output feature.
                This can also be provided once and for all at the instantiation of InformationSource, through the
                default_model_to_split_mapping argument.
            extra: Dictionary containing any additional parameter that should be passed to the signal object.

        Returns:
        -------
            The signal value.

        """
        results = []
        # Compute the signal for each model
        for k, model in enumerate(models):
            # Extract the features to be used
            (
                dataset_index,
                split_name,
                input_feature,
                output_feature,
            ) = model_to_split_mapping[k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)
            y = datasets[dataset_index].get_feature(split_name, output_feature)
            results.append(model.get_grad(x, y))
        return results


########################################################################################################################
# MODEL_HOPSKIPJUMPDISTANCE CLASS
########################################################################################################################

class HopSkipJumpDistance(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the hop skip jump distance of a model.
    """

    def __call__(  # noqa: D102
        self:Self,
        model: Model,
        data_loader: DataLoader,
        logger: logger.Logger,
        norm: int = 2,
        y_target: Optional[int] = None,
        image_target: Optional[int] = None,
        initial_num_evals: int = 100,
        max_num_evals: int = 10000,
        stepsize_search: str = "geometric_progression",
        num_iterations: int = 100,
        gamma: float = 1.0,
        constraint: int = 2,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Built-in call method.

        Args:
        ----
            model: The model to be used.
            data_loader: The data loader to load the data.
            logger: The logger object for logging.
            norm: The norm to be used for distance calculation.
            y_target: The target class label (optional).
            image_target: The target image index (optional).
            initial_num_evals: The initial number of evaluations.
            max_num_evals: The maximum number of evaluations.
            stepsize_search: The step size search strategy.
            num_iterations: The number of iterations.
            gamma: The gamma value.
            constraint: The constraint value.
            batch_size: The batch size.
            verbose: Whether to print verbose output.

        Returns:
        -------
            Tuple containing the perturbed images and perturbed distance.

        """


        # Compute the signal for each model
        perturbed_imgs, perturbed_distance = model.get_hop_skip_jump_distance(
                                                    data_loader,
                                                    logger,
                                                    norm,
                                                    y_target,
                                                    image_target,
                                                    initial_num_evals,
                                                    max_num_evals,
                                                    stepsize_search,
                                                    num_iterations,
                                                    gamma,
                                                    constraint,
                                                    batch_size,
                                                    verbose
                                                    )

        return perturbed_imgs, perturbed_distance
