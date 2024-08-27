"""Parent class for user inputs."""

import logging
from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.dev_utils.data_modules import DataModule
from leakpro.fl_utils.gia_optimizers import MetaOptimizer
from leakpro.import_helper import Self


class AbstractGIAInputHandler(ABC):
    """Parent class for user inputs."""

    def __init__(self:Self, configs: dict, logger:logging.Logger,
                 target_model: Module, data_module: DataModule) -> None:
        self.configs = configs
        self.logger = logger
        self.target_model = target_model
        self.data_module = data_module
        self.client_loader = self.data_module.get_subset(self.configs["audit"]["gia_settings"]["num_client_images"])
        self.at_tensor, self.at_loader = self.data_module.get_at_images(self.client_loader)

    def get_meanstd(self: Self) -> tuple[Tensor, Tensor]:
        """Get mean and std from the data."""
        return self.data_module.get_meanstd()

    def get_client_loader(self: Self) -> DataLoader:
        """Get the client data loader."""
        return self.client_loader

    def get_at_images(self: Self) -> DataLoader:
        """Get attack image loader in the shape of client data."""
        return self.at_tensor, self.at_loader

    @abstractmethod
    def train(self: Self,
        data: DataLoader = None,
        optimizer: MetaOptimizer = None,
        ) -> list:
        """Train a model and return the gradients, without updating the model."""
        pass
