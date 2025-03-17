"""Parent class for user inputs."""

from abc import ABC, abstractmethod

from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from leakpro.schemas import TrainingOutput
from leakpro.utils.import_helper import Any, Dict, Self


class AbstractInputHandler(ABC):
    """Parent class for user inputs."""

    def __init_subclass__(cls, **kwargs:dict) -> None:
        """Enforces that all subclasses must define a nested class named 'UserDataset'."""
        super().__init_subclass__(**kwargs)

        # Check if 'UserDataset' is defined in the subclass
        if not hasattr(cls, "UserDataset") or not issubclass(cls.UserDataset, Dataset):
            raise TypeError(f"Class {cls.__name__} must define a nested class named 'UserDataset'.")


    @abstractmethod
    def train(
        self,
        dataloader: DataLoader,
        model: Module,
        criterion: _Loss,
        optimizer: Optimizer
    ) -> TrainingOutput:
        """Procedure to train a model on data from the population."""
        pass

    class UserDataset(Dataset, ABC):
        """Parent class for user-defined datasets."""

        @abstractmethod
        def __init__(self, data: Any, targets: Any, **kwargs: dict) -> None:
            """Abstract base class for datasets. Must be implemented in subclasses."""
            pass

        def return_params(self:Self) -> Dict[str, Any]:
            """Returns required parameters dynamically."""
            return {k: v for k, v in vars(self).items() if k not in {"data", "targets"}}

        def __len__(self) -> int:
            """Return the length of the dataset."""
            return len(self.targets)

        def __getitem__(self, index:int) -> Any:
            """Return a sample from the dataset."""
            return self.data[index], self.targets[index]
