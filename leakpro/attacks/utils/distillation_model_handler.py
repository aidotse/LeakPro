"""Module for handling shadow models."""

import os
import pickle

import numpy as np
import torch.nn.functional as F  # noqa: N812
from torch import cuda, device, save
from torch.nn import CrossEntropyLoss, KLDivLoss, Module
from tqdm import tqdm

from leakpro.attacks.utils.model_handler import ModelHandler
from leakpro.import_helper import Self
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler


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
class DistillationModelHandler(ModelHandler):
    """A class handling the creation, training, and loading of distillation models."""

    def __init__(self:Self, handler: AbstractInputHandler)->None:
        """Initialize the DistillationModelHandler.

        Args:
        ----
            handler (AbstractInputHandler): The input handler.

        """
        super().__init__(handler)
        self.configs = handler.configs["distillation_model"]

        module_path = self.configs.get("module_path", None)
        model_class_path =  self.configs.get("model_class", None)
        self.storage_path = self.configs.get("storage_path", None)
        self.batch_size = self.configs.get("batch_size", 32)
        self.epochs = self.configs.get("epochs", 10)
        self.optimizer_config = self.configs.get("optimizer", None)
        self.loss_config = self.configs.get("loss", None)

        # If no path to distillation model is provided, use the target model blueprint
        if module_path is None or model_class_path is None:
            self.model_blueprint = None
        else:
            self.init_params = self.configs.get("init_params", {})
            self._import_model_from_path(module_path, model_class_path)

            if self.optimizer_config is None:
                raise ValueError("Optimizer configuration not provided")
            self._get_optimizer_class(self.optimizer_config.pop("name"))

            if self.loss_config is None:
                raise ValueError("Loss configuration not provided")
            self._get_criterion_class(self.loss_config.pop("name"))

        # Check if the folder does not exist
        if not os.path.exists(self.storage_path):
            # Create the folder
            os.makedirs(self.storage_path)
            self.logger.info(f"Created folder {self.storage_path}")

        if self.batch_size < 0:
            raise ValueError("Batch size cannot be negative")

        if self.epochs < 0:
            raise ValueError("Number of epochs cannot be negative")

        self.model_storage_name = "distillation_epochs"
        self.metadata_storage_name = "metadata"

        self.model_pairs = {}

    def add_student_teacher_pair(self:Self, name:str, teacher:Module)->None:
        """Add a student-teacher pair to the model handler.

        Args:
        ----
            name (str): The name of the model pair.
            teacher (Module): The teacher model.

        Returns:
        -------
            None

        """
        student, _, optimizer = self._get_model_criterion_optimizer()
        self.model_pairs[name] = {"student": student, "teacher": teacher, "optimizer": optimizer}

    def distill_model(
        self:Self,
        model_pair_name:str,
        num_trajectory_epochs:int,
        distillation_data_indices: np.ndarray,
        label_only:bool=False
    ) -> list[Module]:
        """Create and train shadow models based on the blueprint.

        Args:
        ----
            model_pair_name (str): The name of the model pair.
            num_trajectory_epochs (int): The number of trajectory epochs for training.
            distillation_data_indices (np.ndarray): The indices of the distillation data.
            label_only (bool): The mode of attack.

        Returns:
        -------
            list[Module]: A list of distillation model checkpoints.

        """
        if num_trajectory_epochs < 0:
            raise ValueError("Number of trajectory epochs cannot be negative")

        model_pair = self.model_pairs[model_pair_name]
        student_model = model_pair["student"]
        teacher_model = model_pair["teacher"]
        optimizer = model_pair["optimizer"] # optimizer for student model

        # Get the device for training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        teacher_model.to(gpu_or_cpu)
        student_model.to(gpu_or_cpu)
        student_model.train()
        teacher_model.eval()

        data_loader = self.handler.get_dataloader(distillation_data_indices, self.batch_size)
        self.logger.info(f"Created distillation dataset with size {len(distillation_data_indices)}")

        distillation_checkpoints = []

        for d in range(num_trajectory_epochs):
            epoch_loss = 0

            # Loop over the training set
            for data, target_labels in tqdm(data_loader, desc=f"Epoch {d+1}/{num_trajectory_epochs}"):

                # Move data to the device
                data = data.to(gpu_or_cpu, non_blocking=True)
                target_labels = target_labels.to(gpu_or_cpu, non_blocking=True).long()

                # Output of the distillation model
                output_student = student_model(data)
                output_teacher = teacher_model(data)

                # TODO: add hopskipjump distance here
                if label_only:
                    loss = CrossEntropyLoss()(output_student, target_labels) # TODO: I think this is wrong
                else:
                    loss = KLDivLoss(reduction="batchmean")(F.log_softmax(output_student, dim=1),
                                                            F.softmax(output_teacher.float(), dim=1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            self.logger.info(f"Epoch {d+1}/{num_trajectory_epochs} | Loss: {epoch_loss}")
            with open(f"{self.storage_path}/{model_pair_name}_{d}.pkl", "wb") as f:
                save(student_model.state_dict(), f)
                self.logger.info(f"Saved distillation model for epoch {d} to {self.storage_path}")
            distillation_checkpoints.append(student_model)

            self.logger.info("Storing metadata for distillation model")
            meta_data = {}
            meta_data["init_params"] = self.init_params
            meta_data["train_indices"] = distillation_data_indices
            meta_data["num_train"] = len(distillation_data_indices)
            meta_data["optimizer"] = optimizer.__class__.__name__
            meta_data["batch_size"] = self.batch_size
            meta_data["epochs"] = self.epochs
            meta_data["label_only"] = label_only

            with open(f"{self.storage_path}/{model_pair_name}_metadata_{d}.pkl", "wb") as f:
                pickle.dump(meta_data, f)

            self.logger.info(f"Metadata for distillation model stored in {self.storage_path}")

        return distillation_checkpoints
