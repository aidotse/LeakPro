import importlib.util
import inspect
import os

import torch
from torch import nn

from leakpro.import_helper import Callable, ModuleType, Self


class ShadowModelHandler():
    """Handles the creation, training, and loading of shadow models."""

    def __init__(self:Self, target_model:nn.Module, target_config:dict, config:dict)->None:
        """Initialize the ShadowModelHandler.

        Args:
        ----
            target_model (nn.Module): The target model.
            target_config (dict): The configuration of the target model.
            config (dict): The configuration of the ShadowModelHandler.

        """
        model_path = config["shadow_model"]["model_path"]
        model_class = config["shadow_model"]["model_class"]

        # If no path to shadow model is provided, use the target model blueprint
        if model_path is None or model_class is None:
            self.init_params = target_config["init_params"]
            self.shadow_model_blueprint = target_model.__class__
        else:
            self.model_path = model_path
            self.model_class = model_class
            self.init_params = config["shadow_model"]["init_params"]
            module = self.import_module_from_file(self.model_path)
            self.shadow_model_blueprint = self.get_class_from_module(module, self.model_class)

        self.storage_path = config["audit"]["attack_folder"]
        # Check if the folder does not exist
        if not os.path.exists(self.storage_path):
            # Create the folder
            os.makedirs(self.storage_path)

    def _import_module_from_file(self:Self, filepath:str) -> ModuleType:
        # Import a module from a given file path
        module_name = filepath.split("/")[-1].split(".")[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _get_class_from_module(self:Self, module:ModuleType, class_name:str) -> Callable:
        # Get the specified class from a module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name == class_name:
                return obj
        raise ValueError(f"Class {class_name} not found in module {module.__name__}")

    def create_shadow_models(self:Self, num_models:int) -> None:
        """Create shadow models based on the blueprint.

        Args:
        ----
            num_models (int): The number of shadow models to create.

        Returns:
        -------
            None

        """
        if num_models < 0:
            raise ValueError("Number of models cannot be negative")

        entries = os.listdir(self.storage_path)
        num_to_reuse = len(entries)

        for i in range(num_to_reuse, num_models):
            shadow_model = self.shadow_model_blueprint(**self.init_params)
            with open(f"{self.storage_path}/shadow_model_{i}.pkl", "wb") as f:
                torch.save(shadow_model.state_dict(), f)

    def load_shadow_model(self:Self, index:int) -> nn.Module:
        """Load a shadow model from a saved state.

        Args:
        ----
            index (int): The index of the shadow model to load.

        Returns:
        -------
            nn.Module: The loaded shadow model.

        """
        if index < 0:
            raise ValueError("Index cannot be negative")
        if index >= len(os.listdir(self.storage_path)):
            raise ValueError("Index out of range")
        shadow_model = self.shadow_model_blueprint(**self.init_params)
        shadow_model.load_state_dict(torch.load(f"{self.storage_path}/shadow_model_{index}.pkl"))
        return shadow_model
