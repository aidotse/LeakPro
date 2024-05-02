"""Module for handling shadow models."""

import logging
import os
import pickle
import re

import joblib
import numpy as np
from torch import cuda, device, load, nn, optim, save, functional
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from leakpro.import_helper import Self, Tuple
from leakpro.model import PytorchModel
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
class DistillationModelHandler():
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

        self.logger = logger

        # If no path to distillation model is provided, use the target model blueprint
        if module_path is None or model_class_path is None:
            self.init_params = target_config["init_params"]
            self.distillation_model_blueprint = target_model.model_obj.__class__

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

        self.model_storage_name = "distillation_epoch"
        self.metadata_storage_name = "metadata"

    def create_distillation_models(
        self:Self,
        num_students:int,
        num_trajectory_epochs:int,
        dataset:Dataset,
        training_fraction:float
    ) -> None:
        """Create and train shadow models based on the blueprint.

        Args:
        ----
            num_models (int): The number of shadow models to create.
            dataset (torch.utils.data.Dataset): The full dataset available for training the shadow models.
            training_fraction (float): The fraction of the dataset to use for training.

        Returns:
        -------
            None

        """
        if num_students < 0:
            raise ValueError("Number of student models cannot be negative")
        if num_trajectory_epochs < 0:
            raise ValueError("Number of trajectory epochs cannot be negative")

        entries = os.listdir(self.storage_path)
        # Define a regex pattern to match files like epoch_{i}.pkl
        pattern = re.compile(rf"^{self.model_storage_name}_\d+\.pkl$")
        model_files = [f for f in entries if pattern.match(f)]
        num_to_reuse = len(model_files)

        # Get the size of the dataset
        distillation_data_size = int(len(dataset)*training_fraction)
        all_index = np.arange(len(dataset))

        if num_trajectory_epochs > num_to_reuse:
            self.logger.info(f"Distillation training for more {num_trajectory_epochs -num_to_reuse} epochs")

            distillation_data_indices = np.random.choice(all_index, distillation_data_size, replace=False)
            distillation_dataset = dataset.subset(distillation_data_indices)
            distillation_train_loader = DataLoader(distillation_dataset, batch_size=self.batch_size, shuffle=True)
            self.logger.info(f"Created distillation dataset {i} with size {len(distillation_dataset)}")


            if num_to_reuse == 0:
                # shadow_model = self.distillation_model_blueprint(**self.init_params)
                distillation_model = self.intilizing_model()
            else :
                # TODO: loading last model and training on it
                shadow_model = self._load_shadow_model(num_to_reuse - 1)

        for i in range(num_to_reuse, num_trajectory_epochs):
            self.logger.info(f"Training trajectory epoch {i}")
            #TODO: loading last model and training on it
            # shadow_model = self.distillation_model_blueprint(**self.init_params)
            shadow_model, train_acc, train_loss = self._train_distillation_model(
                shadow_model, distillation_train_loader, self.optimizer_config, self.loss_config, self.epochs
            )

            self.logger.info(f"Training distillation model epoch {i} complete")
            with open(f"{self.storage_path}/{self.model_storage_name}_{i}.pkl", "wb") as f:
                save(shadow_model.state_dict(), f)
                self.logger.info(f"Saved shadow model {i} to {self.storage_path}")

            self.logger.info(f"Storing metadata for shadow model {i}")
            meta_data = {}
            meta_data["init_params"] = self.init_params
            meta_data["train_indices"] = shadow_data_indices
            meta_data["num_train"] = shadow_data_size
            meta_data["optimizer"] = self.optimizer_class.__name__
            meta_data["criterion"] = self.criterion_class.__name__
            meta_data["batch_size"] = self.batch_size
            meta_data["epochs"] = self.epochs
            meta_data["learning_rate"] = self.optimizer_config["lr"]
            meta_data["weight_decay"] = self.optimizer_config.get("weight_decay", 0.0)
            meta_data["train_acc"] = train_acc
            meta_data["train_loss"] = train_loss

            with open(f"{self.storage_path}/{self.metadata_storage_name}_{i}.pkl", "wb") as f:
                pickle.dump(meta_data, f)

            self.logger.info(f"Metadata for shadow model {i} stored in {self.storage_path}")

    def _train_sitillation_model(
            self:Self,
            target_model:Module,
            distillation_shadow_model:Module,
            distillation_train_loader:DataLoader,
            optimizer_config:dict,
            loss_config:dict,
            epochs:int
    ) -> Tuple[Module, np.ndarray, np.ndarray]:

        path_distillation_models_shadow = f"{self.log_dir}/distillation_models_shadow"
        # Get the device for training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")

        target_model.to(gpu_or_cpu)

        optimizer = self.optimizer_class(shadow_model.parameters(), **optimizer_config)
        criterion = self.criterion_class(**loss_config)

        distillation_shadow_model.to(gpu_or_cpu)
        target_model.to(gpu_or_cpu)

        for d in range(epochs):

            distillation_shadow_model.train()
            target_model.eval()

            epoch_loss = 0

            # Loop over each epoch
            self.logger.info(f" *** Training distillation model epoch: {d}")

            # Loop over the training set
            for data, target_labels in distillation_train_loader:

                # Move data to the device
                data, target_labels = data.to(gpu_or_cpu, non_blocking=True), target_labels.to(gpu_or_cpu, non_blocking=True)  # noqa: PLW2901
                target_labels = target_labels.long()  # noqa: PLW2901

                # Output of the distillation model
                output = distillation_shadow_model(data)
                output_teacher = target_model(data)

                soft_target = torch.zeros_like(output)
                soft_target[torch.arange(soft_target.shape[0]), target_labels] = 1

                if self.attack_mode == "label_only":
                    loss = CrossEntropyLoss()(output, target_labels)
                elif self.attack_mode == "soft_label":
                    loss = KLDivLoss(reduction="batchmean")(functional.log_softmax(output, dim=1),
                                                            functional.softmax(output_teacher.float(),
                                                            dim=1))


                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            torch.save(distillation_shadow_model.state_dict(), f"{path_distillation_models_shadow}/epoch_{d}.pkl")



    def _train_shadow_model(
        self:Self,
        shadow_model:Module,
        train_loader:DataLoader,
        optimizer_config:dict,
        loss_config:dict,
        epochs:int
    ) -> Tuple[Module, np.ndarray, np.ndarray]:
        """Train a shadow model.

        Args:
        ----
            shadow_model (Module): The shadow model to train.
            train_loader (torch.utils.data.DataLoader): The training data loader.
            optimizer_config (dict): The optimizer configuration to use.
            loss_config (dict): The loss function configuration to use.
            epochs (int): The number of epochs to train the model.

        Returns:
        -------
            Tuple[Module, np.ndarray, np.ndarray]: The trained shadow model, the training accuracy, and the training loss.

        """
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")

        shadow_model.to(gpu_or_cpu)
        shadow_model.train()

        optimizer = self.optimizer_class(shadow_model.parameters(), **optimizer_config)
        criterion = self.criterion_class(**loss_config)

        for epoch in range(epochs):
            train_loss, train_acc = 0, 0
            shadow_model.train()
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.long()  # noqa: PLW2901
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)  # noqa: PLW2901
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
        shadow_model.to("cpu")
        return shadow_model, train_acc, train_loss

    def _load_shadow_model(self:Self, index:int) -> Module:
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
        shadow_model = self.distillation_model_blueprint(**self.init_params)
        with open(f"{self.storage_path}/{self.model_storage_name}_{index}.pkl", "rb") as f:
            shadow_model.load_state_dict(load(f))
            self.logger.info(f"Loaded shadow model {index}")
        return PytorchModel(shadow_model, self.criterion_class(**self.loss_config))

    def get_shadow_models(self:Self, num_models:int) -> list:
        """Load the the shadow models."""
        shadow_models = []
        shadow_model_indices = []
        for i in range(num_models):
            self.logger.info(f"Loading shadow model {i}")
            shadow_models.append(self._load_shadow_model(i))
            shadow_model_indices.append(i)
        return shadow_models, shadow_model_indices

    def identify_models_trained_on_samples(self:Self, shadow_model_indices: list[int], sample_indices:set[int]) -> list:
        """Identify the shadow models trained on the provided samples.

        Args:
        ----
            shadow_model_indices (list[int]): The indices of the shadow models.
            sample_indices (set[int]): The indices of the samples.

        Returns:
        -------
            list: The list of shadow models trained on the provided samples.

        """
        if shadow_model_indices is None:
            raise ValueError("Shadow model indices must be provided")
        if sample_indices is None:
            raise ValueError("Sample indices must be provided")

        if isinstance(sample_indices, list):
            sample_indices = set(sample_indices)

        self.logger.info("Identifying shadow models trained on provided samples")
        shadow_model_trained_on_data_index = np.zeros((len(shadow_model_indices), len(sample_indices)), dtype=bool)
        for i in shadow_model_indices:
            with open(f"{self.storage_path}/{self.metadata_storage_name}_{i}.pkl", "rb") as f:
                meta_data = joblib.load(f)
                train_indices = set(meta_data["train_indices"])

                for j in range(len(sample_indices)):
                    shadow_model_trained_on_data_index[i, j] = sample_indices[j] in train_indices

        return shadow_model_trained_on_data_index
