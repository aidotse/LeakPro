# TODO: add abstract parent class, allow loading from outside of the package with importlib.util.spec_from_file_location

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Type, Optional
from leakpro.dataset import GeneralDataset
import logging
from abc import ABC, abstractmethod



class CodeHandler(ABC):

    def __init__(self, configs: dict, logger:logging.Logger):
        self.configs = configs
        self.logger = logger
        self.target_model_blueprint: Optional[Type[torch.nn.Module]] = None
        self.target_model: Optional[torch.nn.Module] = None
        self.target_metadata: Optional[dict] = None
        self.population: Optional[GeneralDataset] = None

    def setup(self) -> None:
        self.get_model_class()
        self.get_target_metadata()
        self.get_trained_target_model()
        self.get_population()

    @abstractmethod
    def get_dataloader(self, indices: np.ndarray) -> DataLoader:
        pass

    @abstractmethod
    def get_model_class(self) -> None:
        pass

    @abstractmethod
    def get_target_metadata(self) -> None:
        pass

    @abstractmethod
    def get_trained_target_model(self) -> None:
        pass

    @abstractmethod
    def get_population(self) -> None:
        pass
    
    @abstractmethod
    def train_model(self):
        pass

    def get_shadow_model_class(self) -> Type[torch.nn.Module]:
        # Class of the shadow models. Returns class of target model by deafult. Can be customized if desired.
        self.logger.info("Shadow model blueprint: target model")
        return self.target_model_blueprint

    def get_shadow_model_init_params(self) -> dict:
        # parameters to initialize the shadow model. By default the same as used for the target model
        return self.target_metadata["model_metadata"]["init_params"]

    @property
    def loss(self) -> nn.modules.loss._Loss:
        return nn.CrossEntropyLoss()

    @property
    def model_class(self) -> Type[torch.nn.Module]:
        return self.target_model_blueprint
    
    @property
    def trained_model(self) -> torch.nn.Module:
        return self.target_model
    
    @property
    def target_metadata(self) -> dict:
        return self.target_metadata
    
    @property
    def population(self) -> GeneralDataset:
        return self.population