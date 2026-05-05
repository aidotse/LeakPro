#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Parent class for user inputs."""

import logging
from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader

from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaOptimizer, MetaSGD
from leakpro.utils.import_helper import Self


class AbstractGIAInputHandler(ABC):
    """Parent class for user inputs."""

    def __init__(self:Self, configs: dict, logger:logging.Logger,
                 target_model: Module, data_module: DataLoader) -> None:
        self.configs = configs
        self.logger = logger
        self.target_model = target_model
        self.data_module = data_module
        self.client_loader = self.data_module.get_subset(self.configs["audit"]["gia_settings"]["num_client_images"])
        self.at_tensor, self.at_loader = self.data_module.get_at_images(self.client_loader)

    def get_criterion(self:Self)->None: # add more criterions and add it to audit
        """Set the CrossEntropyLoss for the model."""
        return CrossEntropyLoss()

    def get_optimizer(self: Self) -> MetaOptimizer:
        """Set the optimizer for the model."""
        optimizer = self.configs["audit"]["gia_settings"]["optimizer"]
        lr = self.configs["audit"]["gia_settings"]["learning_rate"]
        if optimizer == "SGD":
            return MetaSGD(lr=lr)
        if optimizer == "Adam":
            return MetaAdam(lr=lr)
        raise ValueError(f"Optimizer '{optimizer}' not found. Please check the optimizer settings.")

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
