# TODO: add abstract parent class, allow loading from outside of the package with importlib.util.spec_from_file_location

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Type, Optional
from leakpro.utils.input_handler import get_class_from_module, import_module_from_file
from leakpro.dataset import GeneralDataset
import logging
import joblib
from leakpro.user_code.parent_template import CodeHandler



class ExampleCodeHandler(CodeHandler):

    def __init__(self, configs: dict, logger:logging.Logger):
        super().__init__(configs = configs, logger = logger)

    def get_dataloader(self, indices: np.ndarray) -> DataLoader:
        pass

    def get_model_class(self) -> None:
        target_module = import_module_from_file(self.configs["target"]["module_path"])
        target_model_blueprint = get_class_from_module(target_module, self.configs["target"]["model_class"])
        self.logger.info(f"Target model blueprint created from {self.configs['target']['model_class']} in {self.configs['target']['module_path']}")
        self.target_model_blueprint = target_model_blueprint

    def get_target_metadata(self) -> None:
        # there should be more documentation what the metadata actually needs. I was very surprised to find the indices definition in there.
        target_model_metadata_path = self.configs["target"]["trained_model_metadata_path"]
        try:
            with open(target_model_metadata_path, "rb") as f:
                self.target_model_metadata = joblib.load(f)
        except FileNotFoundError:
            self.logger.error(f"Could not find the target model metadata at {target_model_metadata_path}")
    
    def get_trained_model(self) -> torch.nn.Module:
        with open(self.configs["target"]["trained_model_path"], "rb") as f:
            target_model = self.target_model_blueprint(**self.target_model_metadata["model_metadata"]["init_params"])
            target_model.load_state_dict(torch.load(f))

    def get_population(self) -> GeneralDataset:
         # Get the population dataset
        try:
            with open(self.configs["target"]["data_path"], "rb") as file:
                self.population = joblib.load(file)
                self.logger.info(f"Loaded population dataset from {self.configs['target']['data_path']}")
        except FileNotFoundError:
            self.logger.error(f"Could not find the population dataset at {self.configs['target']['data_path']}")
        
    def train_model():
        pass
