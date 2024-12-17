"""Parent class for user inputs."""

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader

from leakpro.utils.import_helper import Self


class AbstractInputHandler(ABC):
    """Parent class for user inputs."""

    def __init__(self:Self, configs: dict) -> None:
        self.configs = configs

    @abstractmethod
    def get_criterion(self:Self, criterion: torch.nn.modules.loss._Loss) -> None:
        """Get the loss function for the target model to be used in model training."""
        pass

    @abstractmethod
    def get_optimizer(self:Self, model:torch.nn.Module) -> torch.optim.Optimizer:
        """Get the optimizer used for the target model to be used in model training."""
        pass

    @abstractmethod
    def train(
        self: Self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer
    ) -> nn.Module:
        """Procedure to train a model on data from the population."""
        pass
