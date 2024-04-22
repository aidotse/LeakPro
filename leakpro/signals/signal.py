"""Signal class, which is an abstract class representing any type of signal that can be obtained."""

from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader

from leakpro.dataset import Dataset
from leakpro.import_helper import List, Self, Tuple
from leakpro.model import Model

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
            datasets: List of datasets that can be queried.

        Returns:
        -------
            The signal value.

        """
        results = []
        for model in models:
            # Initialize a list to store the logits for the current model
            model_logits = []

            # Iterate over the dataset using the DataLoader (ensures we use transforms etc)
            data_loader = DataLoader(datasets, batch_size=len(datasets), shuffle=False)
            for data, labels in data_loader:

                # Get rescaled logits for each data point
                logits = model.get_rescaled_logits(data, labels)
                model_logits.extend(logits)
            model_logits = np.array(model_logits)
            # Append the logits for the current model to the results
            results.append(model_logits)

        return results


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
        datasets: List[Dataset],
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
        results = []
        # Compute the signal for each model
        for k, model in enumerate(models):
            x = datasets[k]._data
            y = datasets[k]._labels

            # Compute the signal for each sample
            results.append(model.get_loss(x, y))
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
# Group Information
########################################################################################################################


class GroupInfo(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the group membership of data records.
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
                model_to_split_mapping[i][2] contains the name of the group feature
                This can also be provided once and for all at the instantiation of InformationSource, through the
                default_model_to_split_mapping argument.
            extra: Dictionary containing any additional parameter that should be passed to the signal object.

        Returns:
        -------
            The signal value.

        """
        results = []
        # Given the group membership for each dataset used by each model
        for k in range(len(models)):
            dataset_index, split_name, group_feature = model_to_split_mapping[k]
            g = datasets[dataset_index].get_feature(split_name, group_feature)
            results.append(g)
        return results


class ModelGradientNorm(Signal):
    """sed to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the gradient norm of a model.
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
            results.append(model.get_gradnorm(x, y))
        return results
