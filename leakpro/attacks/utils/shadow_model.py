"""Module for handling shadow models."""

import logging
import os
import pickle

import numpy as np
from torch import load, save
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset

from leakpro.import_helper import Self, Tuple
from leakpro.utils.input_handler import get_class_from_module, import_module_from_file


def singleton(cls):  # noqa: ANN001, ANN201
    """Decorator to create a singleton with initialization parameters."""
    instances = {}
    params = {}

    def get_instance(*args, **kwargs):  # noqa: ANN003, ANN002, ANN202
        if cls not in instances:
            # Store the initialization parameters when the singleton is first created
            params[cls] = (args, kwargs)
            instances[cls] = cls(*args, **kwargs)  # Create the singleton instance
        elif args or kwargs:
            # Raise an error if trying to reinitialize with different parameters
            raise ValueError("Singleton already created with specific parameters.")
        return instances[cls]

    return get_instance

@singleton
class ShadowModelHandler():
    """A class handling the creation, training, and loading of shadow models."""

    def __init__(self:Self, target_model:Module, target_config:dict, config:dict, logger:logging.Logger)->None:
        """Initialize the ShadowModelHandler.

        Args:
        ----
            target_model (Module): The target model.
            target_config (dict): The configuration of the target model.
            config (dict): The configuration of the ShadowModelHandler.
            logger (logging.Logger): The logger object for logging.

        """
        module_path = config["module_path"]
        model_class_path = config["model_class_path"]

        self.logger = logger

        # If no path to shadow model is provided, use the target model blueprint
        if module_path is None or model_class_path is None:
            self.init_params = target_config["init_params"]
            self.shadow_model_blueprint = target_model.__class__

            self.logger.info("Shadow model blueprint: target model")
        else:
            self.module_path = module_path
            self.model_class_path = model_class_path
            self.init_params = config["init_params"]
            module = import_module_from_file(self.module_path)
            self.shadow_model_blueprint = get_class_from_module(module, self.model_class_path)

            self.logger.info(f"Shadow model blueprint: {self.model_class_path} from {self.module_path}")

        self.storage_path = config["storage_path"]
        # Check if the folder does not exist
        if not os.path.exists(self.storage_path):
            # Create the folder
            os.makedirs(self.storage_path)
            self.logger.info(f"Created folder {self.storage_path}")

        self.batch_size = config.get("batch_size", 64)
        if self.batch_size < 0:
            raise ValueError("Batch size cannot be negative")

        self.epochs = config.get("epochs", 10)
        if self.epochs < 0:
            raise ValueError("Number of epochs cannot be negative")

        self.lr = config.get("lr", 0.001)
        if self.lr < 0:
            raise ValueError("Learning rate cannot be negative")

        self.weight_decay = config.get("weight_decay", 0)
        if self.weight_decay < 0:
            raise ValueError("Weight decay cannot be negative")

        self.model_storage_name = "shadow_model"
        self.metadata_storage_name = "metadata"


    def create_shadow_models(
        self:Self,
        num_models:int,
        dataset:Dataset,
        training_fraction:float,
        optimizer:str,
        criterion:str,
    ) -> None:
        """Create and train shadow models based on the blueprint.

        Args:
        ----
            num_models (int): The number of shadow models to create.
            dataset (torch.utils.data.Dataset): The full dataset available for training the shadow models.
            training_fraction (float): The fraction of the dataset to use for training.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            criterion (Module): The loss function to use for training.

        Returns:
        -------
            None

        """
        if num_models < 0:
            raise ValueError("Number of models cannot be negative")

        entries = os.listdir(self.storage_path)
        num_to_reuse = len(entries)

        # Get the size of the dataset
        shadow_data_size = (len(dataset)*training_fraction).astype(int)
        all_index = np.arange(shadow_data_size)

        for i in range(num_to_reuse, num_models):
            self.logger.info(f"Training shadow model {i}")

            shadow_data_indices = np.random.choice(all_index, shadow_data_size, replace=False)
            shadow_dataset = Subset(dataset, shadow_data_indices)
            shadow_train_loader = DataLoader(shadow_dataset, batch_size=self.batch_size, shuffle=True)
            self.logger.info(f"Created shadow dataset {i} with size {len(shadow_dataset)}")

            shadow_model = self.shadow_model_blueprint(**self.init_params)
            train_acc, train_loss = self._train_shadow_model(shadow_model, shadow_train_loader, optimizer, criterion, self.epochs)

            self.logger.info(f"Training shadow model {i} complete")
            with open(f"{self.storage_path}/{self.model_storage_name}_{i}.pkl", "wb") as f:
                save(shadow_model.state_dict(), f)
                self.logger.info(f"Saved shadow model {i} to {self.storage_path}")

            self.logger.info(f"Storing metadata for shadow model {i}")
            meta_data = {}
            meta_data["init_params"] = self.init_params
            meta_data["train_indices"] = shadow_data_indices
            meta_data["num_train"] = shadow_data_size
            meta_data["optimizer"] = type(optimizer)
            meta_data["criterion"] = type(criterion)
            meta_data["batch_size"] = self.batch_size
            meta_data["epochs"] = self.epochs
            meta_data["learning_rate"] = self.lr
            meta_data["weight_decay"] = self.weight_decay
            meta_data["train_acc"] = train_acc
            meta_data["train_loss"] = train_loss

            with open(f"{self.storage_path}/{self.metadata}_{i}.pkl", "wb") as f:
                pickle.dump(meta_data, f)

            self.logger.info(f"Metadata for shadow model {i} stored in {self.storage_path}")

    def _train_shadow_model(
        self:Self,
        shadow_model:Module,
        train_loader:DataLoader,
        optimizer:Optimizer,
        criterion:Module,
        epochs:int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train a shadow model.

        Args:
        ----
            shadow_model (Module): The shadow model to train.
            train_loader (torch.utils.data.DataLoader): The training data loader.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (Module): The loss function to use.
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
        return train_acc, train_loss

    def load_shadow_model(self:Self, index:int) -> Module:
        """Load a shadow model from a saved state.

        Args:
        ----
            index (int): The index of the shadow model to load.

        Returns:
        -------
            Module: The loaded shadow model.

        """
        if index < 0:
            raise ValueError("Index cannot be negative")
        if index >= len(os.listdir(self.storage_path)):
            raise ValueError("Index out of range")
        shadow_model = self.shadow_model_blueprint(**self.init_params)
        shadow_model.load_state_dict(load(f"{self.storage_path}/shadow_model_{index}.pkl"))
        self.logger.info(f"Loaded shadow model {index}")
        return shadow_model
