import importlib.util
import inspect
import os
import logging

import torch
from torch import nn

from leakpro.import_helper import Callable, ModuleType, Self


class ShadowModelHandler():
    """Handles the creation, training, and loading of shadow models."""

    def __init__(self:Self, target_model:nn.Module, target_config:dict, config:dict, logger:logging.Logger)->None:
        """Initialize the ShadowModelHandler.

        Args:
        ----
            target_model (nn.Module): The target model.
            target_config (dict): The configuration of the target model.
            config (dict): The configuration of the ShadowModelHandler.

        """
        model_path = config["shadow_model"]["model_path"]
        model_class = config["shadow_model"]["model_class"]

        self.logger = logger

        # If no path to shadow model is provided, use the target model blueprint
        if model_path is None or model_class is None:
            self.init_params = target_config["init_params"]
            self.shadow_model_blueprint = target_model.__class__

            self.logger.info("Shadow model blueprint: target model")
        else:
            self.model_path = model_path
            self.model_class = model_class
            self.init_params = config["shadow_model"]["init_params"]
            module = self.import_module_from_file(self.model_path)
            self.shadow_model_blueprint = self.get_class_from_module(module, self.model_class)

            self.logger.info(f"Shadow model blueprint: {self.model_class} from {self.model_path}")

        self.storage_path = config["audit"]["attack_folder"]
        # Check if the folder does not exist
        if not os.path.exists(self.storage_path):
            # Create the folder
            os.makedirs(self.storage_path)
            self.logger.info(f"Created folder {self.storage_path}")

    def _import_module_from_file(self:Self, filepath:str) -> ModuleType:
        # Import a module from a given file path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")
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
                self.logger.info(f"Saved shadow model {i}")

    def _train_shadow_model(
        self:Self,
        shadow_model:nn.Module,
        train_loader:torch.utils.data.DataLoader,
        optimizer:torch.optim.Optimizer,
        criterion:nn.Module,
        epochs:int
    ) -> None:
        """Train a shadow model.

        Args:
        ----
            shadow_model (nn.Module): The shadow model to train.
            train_loader (torch.utils.data.DataLoader): The training data loader.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (nn.Module): The loss function to use.
            epochs (int): The number of epochs to train the model.

        Returns:
        -------
            None

        """
        for epoch in range(epochs):
            train_loss, train_acc = 0, 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = shadow_model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs.data.max(1, keepdim=True)[1]
                loss.backward()
                optimizer.step()

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.data.view_as(pred)).sum()
                train_loss += loss.item()

            log_train_str = (
                f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.8f} | "
                f"Train Acc: {float(train_acc)/len(train_loader.dataset):.8f}")
            self.logger.info(log_train_str)

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
        self.logger.info(f"Loaded shadow model {index}")
        return shadow_model

    def train_shadow_models(
        self:Self,
        train_loader:torch.utils.data.DataLoader,
        optimizer:torch.optim.Optimizer,
        criterion:nn.Module,
        epochs:int
    ) -> None:
        """Train all shadow models.

        Args:
        ----
            train_loader (torch.utils.data.DataLoader): The training data loader.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (nn.Module): The loss function to use.
            epochs (int): The number of epochs to train the model.

        Returns:
        -------
            None

        """
        for i in range(len(os.listdir(self.storage_path))):
            shadow_model = self.load_shadow_model(i)
            self._train_shadow_model(shadow_model, train_loader, optimizer, criterion, epochs)
            with open(f"{self.storage_path}/shadow_model_{i}.pkl", "wb") as f:
                torch.save(shadow_model.state_dict(), f)
                self.logger.info(f"Saved shadow model {i}")
                
                
    def save_model_and_metadata(  # noqa: PLR0913
    model: torch.nn.Module,
    data_split: dict,
    configs: dict,
    train_acc: float,
    test_acc: float,
    train_loss: float,
    test_loss: float,
) -> None:
    """Save the model and metadata.

    Args:
    ----
        model (torch.nn.Module): Trained model.
        data_split (dict): Data split for training and testing.
        configs (dict): Configurations for training.
        train_acc (float): Training accuracy.
        test_acc (float): Testing accuracy.
        train_loss (float): Training loss.
        test_loss (float): Testing loss.

    """
    # Save model and metadata
    model_metadata_dict = {"model_metadata": {}}


    log_dir = configs["run"]["log_dir"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/target_model.pkl", "wb") as f:
        torch.save(model.state_dict(), f)
    meta_data = {}

    meta_data["init_params"] = model.init_params if hasattr(model, "init_params") else {}
    meta_data["train_split"] = data_split["train_indices"]
    meta_data["test_split"] = data_split["test_indices"]
    meta_data["num_train"] = len(data_split["train_indices"])
    meta_data["optimizer"] = configs["train"]["optimizer"]
    meta_data["batch_size"] = configs["train"]["batch_size"]
    meta_data["epochs"] = configs["train"]["epochs"]
    meta_data["learning_rate"] = configs["train"]["learning_rate"]
    meta_data["weight_decay"] = configs["train"]["weight_decay"]
    meta_data["train_acc"] = train_acc
    meta_data["test_acc"] = test_acc
    meta_data["train_loss"] = train_loss
    meta_data["test_loss"] = test_loss
    meta_data["dataset"] = configs["data"]["dataset"]

    model_metadata_dict["model_metadata"] = meta_data
    with open(f"{log_dir}/models_metadata.pkl", "wb") as f:
        pickle.dump(model_metadata_dict, f)
