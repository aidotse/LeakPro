"""Parent class for user inputs."""

from abc import ABC, abstractmethod

from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from leakpro.schemas import TrainingOutput


class AbstractInputHandler(ABC):
    """Parent class for user inputs."""

    def __init__(self, configs: dict) -> None:
        self.configs = configs

    @abstractmethod
    def get_criterion(self, criterion: _Loss) -> _Loss:
        """Get the loss function for the target model to be used in model training."""
        pass

    @abstractmethod
    def get_optimizer(self, model:Module) -> Optimizer:
        """Get the optimizer used for the target model to be used in model training."""
        pass

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

    @abstractmethod
    def dpsgd(
        self,
        dataloader: DataLoader,
        model: Module = None,
        optimizer: Optimizer = None,
    ) -> None:
        """Set the model, optimizer and dataset using DPsgd."""
        pass