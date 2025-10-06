"""Signal class, which is an abstract class representing any type of signal that can be obtained."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.fft import fft
from numpy.linalg import inv, norm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.signals.signal_extractor import Model
from leakpro.signals.utils.get_TS2Vec import get_ts2vec_model
from leakpro.utils.import_helper import List, Optional, Self, Tuple


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

    def _get_model_output(  # noqa: ANN204
        self: Self,
        model: Model,
        handler: AbstractInputHandler,
        indices: np.ndarray,
        data_loader: Optional[DataLoader] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if data_loader is None:
            data_loader = handler.get_dataloader(indices, shuffle=False)    # NOTE: Shuffle must be false to maintain indices order
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"
        model_logits = []
        for data, _ in data_loader:
            # Get logits for each data point
            logits = model.get_logits(data)
            model_logits.extend(logits)
        model_logits = np.array(model_logits)
        return model_logits, data_loader.dataset.targets

class ModelLogits(Signal):
    """Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the output of a model.
    """

    def __call__(
        self: Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        data_loader: Optional[list] = None,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            data_loader: Optional DataLoader to use instead of creating a new one from the handler.

        Returns:
        -------
            The signal value.

        """        # Compute the signal for each model

        # Iterate over the DataLoader (ensures we use transforms etc)
        # NOTE: Shuffle must be false to maintain indices order
        if data_loader is None:
            data_loader = handler.get_dataloader(indices, shuffle=False)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for model in tqdm(models, desc="Getting Model Logits"):
            model_logits, _ = self._get_model_output(model, None, None, data_loader)
            results.append(model_logits)
        return np.array(results)

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
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a list to store the logits for the current model
            model_logits = []

            for data, labels in tqdm(data_loader, desc=f"Getting rescaled logits for model {m+1}/ {len(models)}", leave=False):
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
        data_loader = handler.get_dataloader(indices, shuffle=False)
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

class HopSkipJumpDistance(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the hop skip jump distance of a model.
    """

    def __call__(  # noqa: D102
        self:Self,
        model: Model,
        data_loader: DataLoader,
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
        epsilon_threshold: float = 1e-6,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Built-in call method.

        Args:
        ----
            model: The model to be used.
            data_loader: The data loader to load the data.
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
            epsilon_threshold: The epsilon threshold.
            verbose: Whether to print verbose output.

        Returns:
        -------
            Tuple containing the perturbed images and perturbed distance.

        """


        # Compute the signal for each model
        perturbed_imgs, perturbed_distance = model.get_hop_skip_jump_distance(
                                                    data_loader,
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
                                                    epsilon_threshold,
                                                    verbose
                                                    )

        return perturbed_imgs, perturbed_distance

def get_seasonality_coefficients(Y: np.ndarray) -> np.ndarray:  # noqa: N803
    """Extract seasonality coefficients from a multivariate time series using 2D DFT."""
    Z = fft(Y, axis=2)      # column-wise 1D-DFT over variables  # noqa: N806
    return fft(Z, axis=1)   # row-wise 1D-DFT over horizon

class Seasonality(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the seasonality loss of a time series model.
    We define this as the distance (L2 norm) between the true and predicted values for the seasonality component.
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
        for model in tqdm(models, desc="Getting Model Seasonality loss"):
            model_outputs, model_targets = self._get_model_output(model, handler, indices)

            seasonality_pred = get_seasonality_coefficients(model_outputs)
            seasonality_true = get_seasonality_coefficients(model_targets)
            seasonality_loss = norm(seasonality_true - seasonality_pred, axis=(1, 2))
            results.append(seasonality_loss)
        return np.array(results)

def get_trend_coefficients(
        Y: np.ndarray,  # noqa: N803
        polynomial_degree: int = 4
    ) -> np.ndarray:
    """Extract trend coefficients from a multivariate time series by fitting a polynomial to each variable."""
    horizon = Y.shape[1]
    t = np.arange(horizon) / horizon
    P = np.vander(t, polynomial_degree, increasing=True)    # Vandermonde matrix of specified degree  # noqa: N806
    return inv(P.T @ P) @ P.T @ Y                           # least squares solution to polynomial fit

class Trend(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the trend loss of a time series model.
    We define this as the distance (L2 norm) between the true and predicted values for the trend component.
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
        for model in tqdm(models, desc="Getting Model Trend loss"):
            model_outputs, model_targets = self._get_model_output(model, handler, indices)

            trend_pred = get_trend_coefficients(model_outputs)
            trend_true = get_trend_coefficients(model_targets)
            trend_loss = norm(trend_true - trend_pred, axis=(1, 2))
            results.append(trend_loss)
        return np.array(results)

class MSE(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the Mean Squared Error (MSE) of a time series model.
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
        for model in tqdm(models, desc="Getting Model MSE"):
            model_outputs, model_targets = self._get_model_output(model, handler, indices)
            model_mse_loss = np.mean(np.square(model_outputs - model_targets), axis=(1,2))
            results.append(model_mse_loss)
        return np.array(results)

class MAE(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the Mean Average Error (MAE) of a time series model.
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
            batch_size: Integer to determine batch size for dataloader.

        Returns:
        -------
            The signal value.

        """
        results = []
        for model in tqdm(models, desc="Getting Model MAE"):
            model_outputs, model_targets = self._get_model_output(model, handler, indices)
            model_mae_loss = np.mean(np.abs(model_outputs - model_targets), axis=(1,2))
            results.append(model_mae_loss)
        return np.array(results)

class SMAPE(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the Symmetric Mean Absolute Percentage Error (SMAPE) of a time series model.
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
        for model in tqdm(models, desc="Getting Model SMAPE"):
            model_outputs, model_targets = self._get_model_output(model, handler, indices)

            numerator = np.abs(model_outputs - model_targets)
            denominator = np.abs(model_outputs) + np.abs(model_targets) + 1e-30
            fraction = numerator / denominator
            smape_loss = np.mean(fraction, axis=(1,2))

            results.append(smape_loss)
        return np.array(results)


class RescaledSMAPE(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the Rescaled SMAPE of a time series model.
    We define this by applying a logit-like transformation to the original SMAPE,
    mapping the signal range from [0, 1] to an undbounded scale.
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
        for model in tqdm(models, desc="Getting Model Rescaled SMAPE"):
            model_outputs, model_targets = self._get_model_output(model, handler, indices)

            numerator = np.abs(model_outputs - model_targets)
            denominator = np.abs(model_outputs) + np.abs(model_targets) + 1e-30
            fraction = numerator / denominator
            smape_loss = np.mean(fraction, axis=(1,2))
            rescaled_smape = np.log(smape_loss / (1 - smape_loss + 1e-30))

            results.append(rescaled_smape)
        return np.array(results)

class TS2Vec(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the TS2Vec loss of a time series model.
    We define this as the distance (L2 norm) between the TS2Vec representations of the true and predicted values.

    TS2Vec (https://arxiv.org/abs/2106.10466) is an unsupervised representation learning method for time series.
    Here, we train a TS2Vec model on the shadow population’s target time series.
    This representation model is then used to encode both predicted and true sequences,
    and we compute the per-sample distance between their representations.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        shadow_population_indices: np.ndarray
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            shadow_population_indices: List of indices in population dataset used to train the shadow models.

        Returns:
        -------
            The signal value.

        """
        # Get represenation model
        batch_size = handler.get_dataloader(indices, shuffle=False).batch_size
        ts2vec_model = get_ts2vec_model(handler, shadow_population_indices, batch_size)

        # Get signals
        ts2vec_true = ts2vec_model.encode(
            np.array(handler.population.targets)[indices],
            encoding_window="full_series",
            batch_size=batch_size
        )
        results = []
        for model in tqdm(models, desc="Getting Model TS2Vec loss"):
            model_outputs, _ = self._get_model_output(model, handler, indices)
            ts2vec_pred = ts2vec_model.encode(model_outputs, encoding_window="full_series", batch_size=batch_size)
            ts2vec_loss = norm(ts2vec_true - ts2vec_pred, axis=1)
            results.append(ts2vec_loss)

        return results

SIGNAL_REGISTRY = {
    "ModelLogits": ModelLogits,
    "ModelRescaledLogits": ModelRescaledLogits,
    "ModelLoss": ModelLoss,
    "HopSkipJumpDistance": HopSkipJumpDistance,
    "Seasonality": Seasonality,
    "Trend": Trend,
    "MSE": MSE,
    "MAE": MAE,
    "SMAPE": SMAPE,
    "RescaledSMAPE": RescaledSMAPE,
    "TS2Vec": TS2Vec,
}

def create_signal_instance(signal_name: str) -> Signal:
    """Instantiate and return the signal object corresponding to the given name."""
    try:
        signal_cls = SIGNAL_REGISTRY[signal_name]
    except KeyError as e:
        raise ValueError(f"Unknown signal name: '{signal_name}'") from e
    return signal_cls()
