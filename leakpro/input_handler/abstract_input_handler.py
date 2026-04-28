"""Parent class for user inputs."""

from abc import ABC, abstractmethod

import numpy as np
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.import_helper import Any, Dict, List, Self


class AbstractInputHandler(ABC):
    """Parent class for user inputs."""

    def __init_subclass__(cls: type, **kwargs:dict) -> None:
        """Enforces that all subclasses must define a nested class named 'UserDataset'."""
        super().__init_subclass__(**kwargs)

        # Check if 'UserDataset' is defined in the subclass
        if not hasattr(cls, "UserDataset") or not issubclass(cls.UserDataset, Dataset):
            raise TypeError(f"Class {cls.__name__} must define a nested class named 'UserDataset'.")


    @abstractmethod
    def train(
        self: Self,
        dataloader: DataLoader,
        model: Module,
        criterion: _Loss,
        optimizer: Optimizer
    ) -> TrainingOutput:
        """Procedure to train a model on data from the population."""
        pass

    @abstractmethod
    def eval(
        self: Self,
        dataloader: DataLoader,
        model: Module,
        criterion: _Loss,
        device: str
    ) -> EvalOutput:
        """Procedure to train a model on data from the population."""
        pass

    def sample_shadow_indices(
        self: Self,
        shadow_population: List[int],
        data_fraction: float
    ) -> np.ndarray:
        """Sample indices for shadow model training.

        May be overridden in Handler implementation to perform custom sampling;
        see for example sampling by individual in time series handlers.

        Args:
        ----
            shadow_population: List of available indices in the shadow population.
            data_fraction: Fraction of data to sample.

        Returns:
        -------
            Array of sampled indices.

        """
        data_size = int(len(shadow_population) * data_fraction)
        return np.random.choice(shadow_population, data_size, replace=False)

    class UserDataset(Dataset, ABC):
        """Parent class for user-defined datasets."""

        @abstractmethod
        def __init__(self: Self, data: Any, targets: Any, **kwargs: dict) -> None:
            """Abstract base class for datasets. Must be implemented in subclasses."""
            pass

        def return_params(self:Self) -> Dict[str, Any]:
            """Returns required parameters dynamically."""
            return {k: v for k, v in vars(self).items() if k not in {"data", "targets"}}

        def __len__(self: Self) -> int:
            """Return the length of the dataset."""
            return len(self.targets)

        def __getitem__(self: Self, index:int) -> Any:
            """Return a sample from the dataset."""
            return self.data[index], self.targets[index]
