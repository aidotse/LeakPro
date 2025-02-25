"""Parent class for user inputs."""

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader

from leakpro.input_handler.user_imports import (
    get_optimizer_mapping,
    get_criterion_mapping,
)

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

    # @abstractmethod
    # def get_shadow_model_type(self:Self) -> str:
    #     """Get the type of shadow model to be used in the attack."""
    #     pass

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

    def _get_optimizer_class(self:Self, optimizer_name:str) -> None:
        """Get the optimizer class based on the optimizer name.

        Args:
        ----
            optimizer_name (str): The name of the optimizer.

        """
        try:
            return get_optimizer_mapping()[optimizer_name]
        except Exception as e:
            raise ValueError(f"Failed to create optimizer from {self.optimizer_config['name']}") from e
    
    def _get_criterion_class(self:Self, criterion_name:str)->None:
        """Get the criterion class based on the criterion name.

        Args:
        ----
            criterion_name (str): The name of the criterion.

        """
        try:
            return get_criterion_mapping()[criterion_name]
        except Exception as e:
            raise ValueError(f"Failed to create criterion from {self.criterion_config['name']}") from e
