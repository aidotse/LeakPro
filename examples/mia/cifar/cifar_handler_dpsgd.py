"""Module containing the class to handle the user input for the CIFAR10/100 dataset."""

import os
import pickle

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

import torch
from torch import cuda, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

class CifarInputHandlerDPsgd(AbstractInputHandler):
    """Class to handle the user input for the CIFAR10 and CIFAR100 dataset."""

    def train(
        self: Self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
        dpsgd_metadata_path: str = "./target_dpsgd/dpsgd_dic.pkl",
        virtual_batch_size: int = 16,
    ) -> TrainingOutput:
        """
        Train a DP-SGD compliant model.

        Args:
            dataloader (DataLoader): DataLoader for the training dataset.
            model (torch.nn.Module): The model to be trained.
            criterion (torch.nn.Module): Loss function to optimize.
            optimizer (optim.Optimizer): Optimizer for training.
            epochs (int): Number of training epochs.
            dpsgd_meta_data_path (str): Path to the DP-SGD metadata file containing privacy parameters.
            virtual_batch_size (int): Virtual batch size for DP-SGD training.

        Returns:
            TrainingOutput: Contains the trained model and training metrics, including accuracy and loss history.
        """

        # Get targeted batch_size from the dataloader
        batch_size = dataloader.batch_size
        
        # Get dpsgd flag from the model
        RUN_DPSGD = model.dpsgd

        if RUN_DPSGD:
            # Check if the model is compatible with DP-SGD
            errors = ModuleValidator.validate(model, strict=False)
            if len(errors) > 0:
                logger.info("Model has privacy violations. Fixing...")
                # Use ModuleValidator.fix to fix the models privacy violations
                model = ModuleValidator.fix(model)
                
                # Re-instantiate the optimizer with the fixed model parameters
                if optimizer is None:
                    raise ValueError("Optimizer must be provided for DP-SGD training")
                    
                # Get optimizer class and constructor parameters
                optimizer_class = optimizer.__class__
                # Extract all important parameters from the optimizer
                optimizer_config = {}
                for group in optimizer.param_groups:
                    for key, value in group.items():
                        if key != 'params':  # Skip the parameters
                            optimizer_config[key] = value
                
                # Create new optimizer with the same configuration but updated model parameters
                optimizer = optimizer_class(model.parameters(), **optimizer_config)
                logger.info(f"Model fixed and {optimizer_class.__name__} re-instantiated.")

            # Send the model, optimizer, and dataloader to be DP-sgd-compliant 
            model, optimizer, dataloader, _ = dpsgd(
                                                model,
                                                optimizer,
                                                dataloader,
                                                dpsgd_path = dpsgd_metadata_path
                                                )

        # read hyperparams for training (the parameters for the dataloader are defined in get_dataloader):
        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        device = torch.device("cuda" if cuda.is_available() else "cpu")
        model.to(device)

        accuracy_history = []
        loss_history = []

        if RUN_DPSGD:
            logger.info("Training with DP-SGD")

            # Use BatchMemoryManager to handle large batch sizes by splitting them into smaller sub-batches
            # Helpful with limited-memory hardware while maintaining the same overall batch size
            with  BatchMemoryManager(
                data_loader=dataloader, 
                max_physical_batch_size=virtual_batch_size, # Set max pyhsical batch size
                optimizer=optimizer
            ) as dataloader:

                for epoch in range(epochs):
                    train_loss, train_acc = train_loop(
                                dataloader,
                                model,
                                criterion,
                                optimizer,
                                device,
                                epoch,
                                epochs,
                                )

                    avg_train_loss = train_loss / len(dataloader.dataset)
                    train_accuracy = train_acc / len(dataloader.dataset) 
                    accuracy_history.append(train_accuracy) 
                    loss_history.append(avg_train_loss)
                
        else:
            logger.info("Training without DP-SGD")
            for epoch in range(epochs):
                train_loss, train_acc = train_loop(
                                        dataloader,
                                        model,
                                        criterion,
                                        optimizer,
                                        device,
                                        epoch,
                                        epochs
                                        )

                avg_train_loss = train_loss / len(dataloader.dataset)
                train_accuracy = train_acc / len(dataloader.dataset) 
                accuracy_history.append(train_accuracy) 
                loss_history.append(avg_train_loss)

        model.to("cpu")
        
        # Remove the GradSampleModule wrapper.
        if hasattr(model, '_module'):
            model = model._module

        results = EvalOutput(accuracy = train_accuracy,
                             loss = avg_train_loss,
                             extra = {"accuracy_history": accuracy_history, "loss_history": loss_history})
        return TrainingOutput(model = model, metrics=results)

    def eval(self, loader, model, criterion):
        gpu_or_cpu = torch.device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc = 0, 0
        total_samples = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(gpu_or_cpu), target.to(gpu_or_cpu)
                target = target.view(-1) 
                output = model(data)
                loss += criterion(output, target).item() * target.size(0)
                pred = output.argmax(dim=1) 
                acc += pred.eq(target).sum().item()
                total_samples += target.size(0)
            loss /= total_samples
            acc = float(acc) / total_samples
            
        output_dict = {"accuracy": acc, "loss": loss}
        return EvalOutput(**output_dict)

    class UserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, data, targets, **kwargs):
            """
            Args:
                data (Tensor): Image data of shape (N, H, W, C) or (N, C, H, W)
                               Expected to be in range [0,1] (normalized).
                targets (Tensor): Corresponding labels.
                mean (Tensor, optional): Precomputed mean for normalization.
                std (Tensor, optional): Precomputed std for normalization.
            """
            assert data.shape[0] == targets.shape[0], "Data and targets must have the same length"
            assert data.max() <= 1.0 and data.min() >= 0.0, "Data should be in range [0,1]"

            self.data = data.float()  # Ensure float type
            self.targets = targets

            for key, value in kwargs.items():
                setattr(self, key, value)
                
            if not hasattr(self, "mean") or not hasattr(self, "std"):
                # Reshape to (C, 1, 1) for broadcasting
                self.mean = self.data.mean(dim=(0, 2, 3)).view(-1, 1, 1)
                self.std = self.data.std(dim=(0, 2, 3)).view(-1, 1, 1)

        def transform(self, x):
            """Normalize using stored mean and std."""
            return (x - self.mean) / self.std 

        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.targets)

def train_loop(dataloader, model, criterion, optimizer, device, epoch, epochs):

    train_loss, train_acc = 0, 0
    for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
        labels = labels.long()
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        model.zero_grad()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = outputs.argmax(dim=1) 
        loss.backward()
        optimizer.step()

        # Accumulate performance of shadow model
        train_acc += pred.eq(labels.view_as(pred)).sum().item()
        train_loss += loss.item()
    
    return train_acc, train_loss

def dpsgd(
        model: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        dataloader: DataLoader = None,
        dpsgd_path: str = "./target_dpsgd/dpsgd_dic.pkl",
    ) -> None:
    """Set the model, optimizer and dataset using DPsgd."""

    logger.info("Training with DP-SGD")

    sample_rate = 1/len(dataloader)
    # Check if the file exists
    if os.path.exists(dpsgd_path):
        # Open and read the pickle file
        with open(dpsgd_path, "rb") as file:
            privacy_engine_dict = pickle.load(file)
        logger.info("Pickle file loaded successfully, using DPsgd config:", privacy_engine_dict)
    else:
        raise Exception(f"File not found at: {dpsgd_path}")

    try:
        noise_multiplier = get_noise_multiplier(target_epsilon = privacy_engine_dict["target_epsilon"],
                                        target_delta = privacy_engine_dict["target_delta"],
                                        sample_rate = sample_rate ,
                                        epochs = privacy_engine_dict["epochs"],
                                        epsilon_tolerance = privacy_engine_dict["epsilon_tolerance"],
                                        accountant = "prv",
                                        eps_error = privacy_engine_dict["eps_error"],)
    except Exception as e:
        raise ValueError(
            f"Failed to compute noise multiplier using the 'prv' accountant. "
            f"This may be due to a large target_epsilon ({privacy_engine_dict['target_epsilon']}). "
            f"Consider reducing epsilon or switching to a different accountant (e.g., 'rdp'). "
            f"Original error: {e}")

    # make the model private
    privacy_engine = PrivacyEngine(accountant = "prv")
    priv_model, priv_optimizer, priv_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm= privacy_engine_dict["max_grad_norm"]
    )

    return priv_model, priv_optimizer, priv_dataloader, privacy_engine