"""Module for handling shadow models."""

import logging
import os
import pickle

import numpy as np
import torch.nn.functional as F  # noqa: N812
from torch import cuda, device, load, nn, optim, save
from torch.nn import CrossEntropyLoss, KLDivLoss, Module
from torch.utils.data import DataLoader, Dataset

from leakpro.import_helper import Self
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
class DistillationTargetModelHandler():
    """A class handling the creation, training, and loading of distillation models."""

    # Create a dictionary mapping lowercase names to optimizer classes (Optimizer is the base class)
    optimizer_mapping = {
        attr.lower(): getattr(optim, attr)
        for attr in dir(optim)
        if isinstance(getattr(optim, attr), type) and issubclass(getattr(optim, attr), optim.Optimizer)
    }

    # Create a dictionary mapping lowercase names to loss classes (_Loss is the base class)
    loss_mapping = {}

    for attr in dir(nn):
        # Get the attribute
        attribute = getattr(nn, attr, None)
        # Ensure it's a class and a subclass of _Loss
        if isinstance(attribute, type) and issubclass(attribute, nn.modules.loss._Loss):
            loss_mapping[attr.lower()] = attribute

    def __init__(self:Self, target_model:Module, target_config:dict, config:dict, logger:logging.Logger)->None:
        """Initialize the DistillationModelHandler.

        Args:
        ----
            target_model (Module): The target model.
            target_config (dict): The configuration of the target model.
            config (dict): The configuration of the DistillationModelHandler.
            logger (logging.Logger): The logger object for logging.

        """
        module_path = config.get("module_path")
        model_class_path =  config.get("model_class_path")
        self.target_model = target_model
        self.logger = logger

        # If no path to distillation model is provided, use the target model blueprint
        if module_path is None or model_class_path is None:
            self.init_params = target_config["init_params"]
            self.distillation_model_blueprint = self.target_model.model_obj.__class__

            self.logger.info("Distillation model blueprint: target model")
        else:
            self.module_path = module_path
            self.model_class_path = model_class_path
            self.init_params = config.get("init_params", {})
            module = import_module_from_file(self.module_path)
            self.distillation_model_blueprint = get_class_from_module(module, self.model_class_path)

            self.logger.info(f"Distillation model blueprint loaded from {self.model_class_path} from {self.module_path}")

        self.storage_path = config["storage_path"]
        # Check if the folder does not exist
        if not os.path.exists(self.storage_path):
            # Create the folder
            os.makedirs(self.storage_path)
            self.logger.info(f"Created folder {self.storage_path}")

        self.batch_size = config.get("batch_size", target_config["batch_size"])
        if self.batch_size < 0:
            raise ValueError("Batch size cannot be negative")
        #TODO: epoch here?
        self.epochs = config.get("epochs", target_config["epochs"])
        if self.epochs < 0:
            raise ValueError("Number of epochs cannot be negative")

        self.optimizer_config = config.get("optimizer", target_config["optimizer"])
        if self.optimizer_config is None:
            raise ValueError("Optimizer configuration not provided")

        self.loss_config = config.get("loss", target_config["loss"])
        if self.loss_config is None:
            raise ValueError("Loss configuration not provided")

        self.optimizer_class = self.optimizer_mapping[self.optimizer_config.pop("name")]
        self.criterion_class = self.loss_mapping[self.loss_config.pop("name")]


        self.model_storage_name = "distillation_epochs"
        self.metadata_storage_name = "metadata"

    def create_distillation_models(
        self:Self,
        num_students:int,
        num_trajectory_epochs:int,
        dataset:Dataset,
        distillation_data_indices: np.ndarray,
        attack_mode:str
    ) -> None:
        """Create and train shadow models based on the blueprint.

        Args:
        ----
            num_students (int): The number of student models to create.
            num_trajectory_epochs (int): The number of trajectory epochs for training.
            dataset (torch.utils.data.Dataset): The full dataset available for training the shadow models.
            distillation_data_indices (np.ndarray): The indices of the distillation data.
            attack_mode (str): The mode of attack.

        Returns:
        -------
            None

        """
        if num_students < 0:
            raise ValueError("Number of student models cannot be negative")
        if num_trajectory_epochs < 0:
            raise ValueError("Number of trajectory epochs cannot be negative")

        #Intiate the distillation model
        distillation_model = self.distillation_model_blueprint(**self.init_params)


        self.logger.info(f"Distillation training for  {num_trajectory_epochs } epochs")

        self._train_distillation_model(distillation_model,
                                        dataset,
                                        distillation_data_indices,
                                        num_trajectory_epochs,
                                        attack_mode,
            )


    def _train_distillation_model(
            self:Self,
            distillation_model:Module,
            distillation_dataset:Dataset,
            distillation_data_indices:np.ndarray,
            num_trajectory_epochs:int,
            attack_mode:str
    ) -> None:

        # Get the device for training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        distillation_model.to(gpu_or_cpu)
        teacher_model = self.target_model.model_obj
        teacher_model.to(gpu_or_cpu)

        # Data prepration
        distillation_train_loader = DataLoader(distillation_dataset, batch_size=self.batch_size, shuffle=True)
        self.logger.info(f"Created distillation dataset with size {len(distillation_dataset)}")

        optimizer = self.optimizer_class(distillation_model.parameters(), **self.optimizer_config)

        for d in range( num_trajectory_epochs):

            distillation_model.train()
            teacher_model.eval()

            epoch_loss = 0

            # Loop over each epoch
            self.logger.info(f" *** Training distillation model epoch: {d}")

            # Loop over the training set
            for data, target_labels in distillation_train_loader:

                # Move data to the device
                data, target_labels = data.to(gpu_or_cpu, non_blocking=True), target_labels.to(gpu_or_cpu, non_blocking=True)  # noqa: PLW2901
                target_labels = target_labels.long()  # noqa: PLW2901

                # Output of the distillation model
                output = distillation_model(data)
                output_teacher = teacher_model(data)


                if attack_mode == "label_only":
                    loss = CrossEntropyLoss()(output, target_labels)
                elif attack_mode == "soft_label":
                    loss = KLDivLoss(reduction="batchmean")(F.log_softmax(output, dim=1),
                                                            F.softmax(output_teacher.float(),
                                                            dim=1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()


            self.logger.info(f"Training distillation model epoch {d} completed")
            self.logger.info(f"loss: {epoch_loss}")
            with open(f"{self.storage_path}/{self.model_storage_name}_{d}.pkl", "wb") as f:
                save(distillation_model.state_dict().copy(), f)
            self.logger.info(f"Saved distillation model epoch {d} to {self.storage_path}")

            self.logger.info("Storing metadata for distillation model")
            meta_data = {}
            meta_data["init_params"] = self.init_params
            meta_data["train_indices"] = distillation_data_indices
            meta_data["num_train"] = len(distillation_data_indices)
            meta_data["optimizer"] = self.optimizer_class.__name__
            meta_data["criterion"] = self.criterion_class.__name__
            meta_data["batch_size"] = self.batch_size
            #TODO: epoch here?
            meta_data["epochs"] = self.epochs
            meta_data["learning_rate"] = self.optimizer_config["lr"]
            meta_data["weight_decay"] = self.optimizer_config.get("weight_decay", 0.0)

            with open(f"{self.storage_path}/{self.metadata_storage_name}_{d}.pkl", "wb") as f:
                pickle.dump(meta_data, f)

            self.logger.info(f"Metadata for distillation model stored in {self.storage_path}")


    def _load_distillation_epoch(self:Self, index:int) -> Module:
        if index < 0:
            raise ValueError("Index cannot be negative")
        if index >= len(os.listdir(self.storage_path)):
            raise ValueError("Index out of range")
        distillation_epoch = self.distillation_model_blueprint(**self.init_params)
        with open(f"{self.storage_path}/{self.model_storage_name}_{index}.pkl", "rb") as f:
            distillation_epoch.load_state_dict(load(f))
            self.logger.info(f"Loaded distillaion epoch  {index}")
        return distillation_epoch


    def get_distillation_epochs(self:Self, num_epochs:int) -> list:
        """Retrieves the distillation epochs.

        Args:
        ----
            num_epochs (int): The number of epochs to retrieve.

        Returns:
        -------
            list: A list of distillation epochs.

        """
        distillation_epochs = []
        for i in range(num_epochs):
            self.logger.info(f"Loading distillation epoch {i}")
            epoch = self._load_distillation_epoch(i)
            distillation_epochs.append(epoch)

        return distillation_epochs


@singleton
class DistillationShadowModelHandler():
    """A class handling the creation, training, and loading of distillation models."""

    # Create a dictionary mapping lowercase names to optimizer classes (Optimizer is the base class)
    optimizer_mapping = {
        attr.lower(): getattr(optim, attr)
        for attr in dir(optim)
        if isinstance(getattr(optim, attr), type) and issubclass(getattr(optim, attr), optim.Optimizer)
    }

    # Create a dictionary mapping lowercase names to loss classes (_Loss is the base class)
    loss_mapping = {}

    for attr in dir(nn):
        # Get the attribute
        attribute = getattr(nn, attr, None)
        # Ensure it's a class and a subclass of _Loss
        if isinstance(attribute, type) and issubclass(attribute, nn.modules.loss._Loss):
            loss_mapping[attr.lower()] = attribute

    def __init__(self:Self,  config:dict, logger:logging.Logger)->None:
        self.logger = logger
        self.module_path = config.get("module_path")
        self.config = config


    def initializng_shadow_teacher(self:Self,
                                   target_model:Module,
                                   target_metadata:dict)->None:
        """Initialize the shadow teacher model.

        Args:
        ----
            target_model (Module): The target model.
            target_metadata (dict): The metadata of the target model.

        Returns:
        -------
            None

        """
        model_class_path =  self.config.get("model_class_path")
        self.target_model = target_model

        # If no path to distillation model is provided, use the target model blueprint
        if self.module_path is None or model_class_path is None:
            self.init_params = target_metadata["init_params"]
            self.distillation_model_blueprint = target_model.model_obj.__class__

            self.logger.info("Distillation model blueprint: target model")
        else:
            self.init_params = self.config.get("init_params", {})
            module = import_module_from_file(self.module_path)
            self.distillation_model_blueprint = get_class_from_module(module, model_class_path)

            self.logger.info(f"Distillation model blueprint loaded from {model_class_path} from {self.module_path}")

        self.storage_path = self.config["storage_path"]
        # Check if the folder does not exist
        if not os.path.exists(self.storage_path):
            # Create the folder
            os.makedirs(self.storage_path)
            self.logger.info(f"Created folder {self.storage_path}")

        self.batch_size = self.config.get("batch_size", target_metadata["batch_size"])
        if self.batch_size < 0:
            raise ValueError("Batch size cannot be negative")
        #TODO: epoch here?
        self.epochs = self.config.get("epochs", target_metadata["epochs"])
        if self.epochs < 0:
            raise ValueError("Number of epochs cannot be negative")

        self.optimizer_config = self.config.get("optimizer", target_metadata["optimizer"].lower())
        if self.optimizer_config is None:
            raise ValueError("Optimizer configuration not provided")

        self.loss_config = {"name": self.config.get("criterion", target_metadata["criterion"].lower())}
        if self.loss_config is None:
            raise ValueError("Loss configuration not provided")

        self.optimizer_class = self.optimizer_mapping[self.optimizer_config.pop("name")]
        self.criterion_class = self.loss_mapping[self.loss_config.pop("name")]


        self.model_storage_name = "distillation_epochs"
        self.metadata_storage_name = "metadata"

    def create_distillation_models(
        self:Self,
        num_students:int,
        num_trajectory_epochs:int,
        dataset:Dataset,
        distillation_data_indices:np.ndarray,
        attack_mode:str
    ) -> None:
        """Create and train shadow models based on the blueprint.

        Args:
        ----
            num_students (int): The number of student models to create.
            num_trajectory_epochs (int): The number of trajectory epochs for training.
            dataset (torch.utils.data.Dataset): The full dataset available for training the shadow models.
            distillation_data_indices (np.ndarray): The indices of the data points from the dataset to be used for distillation.
            attack_mode (str): The mode of attack.

        Returns:
        -------
            None

        """
        if num_students < 0:
            raise ValueError("Number of student models cannot be negative")
        if num_trajectory_epochs < 0:
            raise ValueError("Number of trajectory epochs cannot be negative")

        #Intiate the distillation model
        distillation_model = self.distillation_model_blueprint(**self.init_params)

        self._train_distillation_model(distillation_model,
                                        dataset,
                                        distillation_data_indices,
                                        num_trajectory_epochs,
                                        attack_mode,
            )


    def _train_distillation_model(
            self:Self,
            distillation_model:Module,
            distillation_dataset:Dataset,
            distillation_data_indices: np.ndarray,
            num_trajectory_epochs:int,
            attack_mode:str
    ) -> None:

        # Get the device for training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        distillation_model.to(gpu_or_cpu)
        teacher_model = self.target_model.model_obj
        teacher_model.to(gpu_or_cpu)

        distillation_train_loader = DataLoader(distillation_dataset, batch_size=self.batch_size, shuffle=True)
        self.logger.info(f"Created distillation dataset with size {len(distillation_dataset)}")

        optimizer = self.optimizer_class(distillation_model.parameters(), **self.optimizer_config)

        for d in range(num_trajectory_epochs):

            distillation_model.train()
            teacher_model.eval()

            epoch_loss = 0

            # Loop over each epoch
            self.logger.info(f" *** Training distillation of shadow model epoch: {d}")

            # Loop over the training set
            for data, target_labels in distillation_train_loader:

                # Move data to the device
                data, target_labels = data.to(gpu_or_cpu, non_blocking=True), target_labels.to(gpu_or_cpu, non_blocking=True)  # noqa: PLW2901
                target_labels = target_labels.long()  # noqa: PLW2901

                # Output of the distillation model
                output = distillation_model(data)
                output_teacher = teacher_model(data)


                if attack_mode == "label_only":
                    loss = CrossEntropyLoss()(output, target_labels)
                elif attack_mode == "soft_label":
                    loss = KLDivLoss(reduction="batchmean")(F.log_softmax(output, dim=1),
                                                            F.softmax(output_teacher.float(),
                                                            dim=1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()


            self.logger.info(f"Training distillation of shadow model epoch {d} completed")
            self.logger.info(f"loss: {epoch_loss}")
            with open(f"{self.storage_path}/{self.model_storage_name}_{d}.pkl", "wb") as f:
                save(distillation_model.state_dict().copy(), f)
                self.logger.info(f"Saved distillation of shadow model epoch {d} to {self.storage_path}")


            self.logger.info("Storing metadata for distillation of shadow model")
            meta_data = {}
            meta_data["init_params"] = self.init_params

            meta_data["train_indices"] = distillation_data_indices
            meta_data["num_train"] = len(distillation_data_indices)
            meta_data["optimizer"] = self.optimizer_class.__name__
            meta_data["criterion"] = self.criterion_class.__name__
            meta_data["batch_size"] = self.batch_size
            #TODO: epoch here?
            meta_data["epochs"] = self.epochs
            meta_data["learning_rate"] = self.optimizer_config["lr"]
            meta_data["weight_decay"] = self.optimizer_config.get("weight_decay", 0.0)
            with open(f"{self.storage_path}/{self.metadata_storage_name}_{d}.pkl", "wb") as f:
                pickle.dump(meta_data, f)

            self.logger.info(f"Metadata for distillation of shadow model stored in {self.storage_path}")


    def _load_distillation_epoch(self:Self, index:int) -> Module:
        if index < 0:
            raise ValueError("Index cannot be negative")
        if index >= len(os.listdir(self.storage_path)):
            raise ValueError("Index out of range")
        distillation_epoch = self.distillation_model_blueprint(**self.init_params)
        with open(f"{self.storage_path}/{self.model_storage_name}_{index}.pkl", "rb") as f:
            distillation_epoch.load_state_dict(load(f))
            self.logger.info(f"Loaded distillaion of shadow model, epoch {index}")
        return distillation_epoch

    def get_distillation_epochs(self:Self, num_epochs:int) -> list:
        """Retrieves the distillation epochs.

        Args:
        ----
            num_epochs (int): The number of epochs to retrieve.

        Returns:
        -------
            list: A list of distillation epochs.

        """
        distillation_epochs = []
        for i in range(num_epochs):
            self.logger.info(f"Loading distillation epoch {i}")
            epoch = self._load_distillation_epoch(i)
            distillation_epochs.append(epoch)
        return distillation_epochs
