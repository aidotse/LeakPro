"""Module for handling shadow models."""

import pickle

import numpy as np
import torch.nn.functional as F  # noqa: N812
from torch import cat, cuda, device, save, sigmoid
from torch.nn import CrossEntropyLoss, KLDivLoss, Module
from tqdm import tqdm

from leakpro.attacks.utils.model_handler import ModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.schemas import DistillationModelTrainingSchema
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


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
            instances[cls].__init__(*args, **kwargs)  # Update the singleton instance with new parameters
        return instances[cls]

    return get_instance

@singleton
class DistillationModelHandler(ModelHandler):
    """A class handling the creation, training, and loading of distillation models."""

    def __init__(self:Self, handler: MIAHandler)->None:
        """Initialize the DistillationModelHandler.

        Args:
        ----
            handler (MIAHandler): The input handler.

        """
        caller = "distillation_model"
        super().__init__(handler, caller)
        self.configs = handler.configs.distillation_model

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

    def distill_model(  # noqa: PLR0915
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

        data_loader = self.handler.get_dataloader(distillation_data_indices)
        logger.info(f"Created distillation dataset with size {len(distillation_data_indices)}")

        distillation_checkpoints = []

        for d in range(num_trajectory_epochs):
            epoch_loss = 0

            # Loop over the training set
            for data, target_labels in tqdm(data_loader, desc=f"Epoch {d+1}/{num_trajectory_epochs}"):

                # Move data to the device
                data = data.to(gpu_or_cpu, non_blocking=True)
                target_labels = target_labels.to(gpu_or_cpu, non_blocking=True)

                # Output of the distillation model
                output_student = student_model(data)
                output_teacher = teacher_model(data)

                # In case we are dealing with binary classification, we need to add a dimension
                if output_student.shape[1] == 1:
                    prob_student_positives = sigmoid(output_student)
                    neg_prob_student = 1 - prob_student_positives
                    prob_teacher_positives = sigmoid(output_teacher)
                    neg_prob_teacher = 1 - prob_teacher_positives
                    student_signal = cat((neg_prob_student, prob_student_positives), dim=1)
                    teacher_signal = cat((neg_prob_teacher, prob_teacher_positives), dim=1)
                else:
                    student_signal = F.log_softmax(output_student, dim=1)
                    teacher_signal = F.softmax(output_teacher.float(), dim=1)

                # TODO: add hopskipjump distance here
                if label_only:
                    loss = CrossEntropyLoss()(output_student, target_labels) # TODO: I think this is wrong (teacher?)
                else:
                    loss = KLDivLoss(reduction="batchmean")(student_signal, teacher_signal)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            logger.info(f"Epoch {d+1}/{num_trajectory_epochs} | Loss: {epoch_loss}")
            with open(f"{self.storage_path}/{model_pair_name}_{d}.pkl", "wb") as f:
                save(student_model.state_dict(), f)
                logger.info(f"Saved distillation model for epoch {d} to {self.storage_path}")
            distillation_checkpoints.append(student_model)

            logger.info("Storing metadata for distillation model")
            meta_data = {
                "init_params" : self.init_params,
                "train_indices" : distillation_data_indices,
                "num_train" : len(distillation_data_indices),
                "optimizer" : optimizer.__class__.__name__,
                "epochs" : self.epochs,
                "label_only" : label_only
            }

            validated_meta_data = DistillationModelTrainingSchema(**meta_data)

            with open(f"{self.storage_path}/{model_pair_name}_metadata_{d}.pkl", "wb") as f:
                pickle.dump(validated_meta_data, f)

            logger.info(f"Metadata for distillation model stored in {self.storage_path}")

        return distillation_checkpoints
