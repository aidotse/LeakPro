"""Parent class for user inputs."""

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from leakpro.fl_utils.gia_optimizers import MetaOptimizer
from leakpro.utils.import_helper import Self


class AbstractGIAInputHandler(ABC):
    """Parent class for user inputs."""

    def __init__(self:Self, configs: dict) -> None:
        self.configs = configs

    @abstractmethod
    def get_criterion(self:Self)->None: # add more criterions and add it to audit
        """Set the CrossEntropyLoss for the model."""
        pass

    @abstractmethod
    def get_optimizer(self: Self) -> MetaOptimizer:
        """Set the optimizer for the model."""
        pass

    @abstractmethod
    def get_client_dataloader(self: Self) -> DataLoader:
        """Get the client dataloader."""
        pass

    @abstractmethod
    def train(self: Self,
        data: DataLoader = None,
        optimizer: MetaOptimizer = None,
        ) -> list:
        """Train a model and return the gradients, without updating the model.

        Note: this must utilize the MetaOptimizer to keep the computational graph intact.
        """
        pass
