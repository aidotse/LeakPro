"""Parent class for user inputs."""

import logging
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from leakpro.fl_utils.gia_optimizers import MetaOptimizer
from leakpro.import_helper import Self, Tuple


class AbstractGIAInputHandler(ABC):
    """Parent class for user inputs."""

    def __init__(self:Self, configs: dict, logger:logging.Logger, client_data: DataLoader, target_model: torch.nn.Module, data_mean, data_std, at_image) -> None:
        self.configs = configs
        self.logger = logger
        self.client_data = client_data
        self.target_model = target_model
        self.data_mean = data_mean
        self.data_std = data_std
        self.at_image = at_image

    def init_at_image(self: Self) -> torch.Tensor:
        """Initializes images with random pixels."""
        # img_shape = self.client_data.dataset[0][0].shape
        # num_images = len(self.client_data.dataset)
        # return torch.randn((num_images, *img_shape),
        #                    **{"device": next(self.target_model.parameters()).device,
        #                        "dtype": next(self.target_model.parameters()).dtype}
        #                        )
        return self.at_image

    def get_meanstd(self: Self) -> Tuple[list, list]:
        """Calculate mean and std of the client dataloader."""
        # dataset = self.client_data.dataset
        # cc = torch.cat([dataset[i][0].reshape(3, -1) for i in range(len(dataset))], dim=1)
        # data_mean = torch.mean(cc, dim=1).tolist()
        # data_std = torch.std(cc, dim=1).tolist()
        # data_mean = torch.as_tensor(data_mean)[:, None, None].to(next(self.target_model.parameters()).device)
        # data_std = torch.as_tensor(data_std)[:, None, None].to(next(self.target_model.parameters()).device)
        return self.data_mean, self.data_std

    @abstractmethod
    def train(self: Self,
        data: DataLoader = None,
        optimizer: MetaOptimizer = None,
        ) -> list:
        """Train a model and return the gradients, without updating the model."""
        pass
