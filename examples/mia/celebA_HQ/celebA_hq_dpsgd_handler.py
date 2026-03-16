"""Input handler for CelebA-HQ experiments with optional DP-SGD training."""

import os
import pickle
from pathlib import Path

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

import torch
from torch import cuda, device, optim, cat
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

class CelebAInputHandlerDPsgd(AbstractInputHandler):
    """Input handler for CelebA-HQ image classification experiments."""

    def _resolve_dpsgd_metadata_path(self: Self, explicit_path: str | None = None) -> str:
        """Resolve the DP-SGD metadata path from arg or target config.

        Resolution order:
        1. Explicit function argument.
        2. `target.dpsgd_path` from config.
        3. `target.target_folder/dpsgd_dic.pkl` fallback.
        """
        if explicit_path:
            return explicit_path

        target_cfg = getattr(getattr(self, "configs", None), "target", None)
        if target_cfg is not None:
            target_path = getattr(target_cfg, "dpsgd_path", None)
            if target_path:
                return target_path

            target_folder = getattr(target_cfg, "target_folder", None)
            if target_folder:
                return str(Path(target_folder) / "dpsgd_dic.pkl")

        raise FileNotFoundError(
            "DP-SGD is enabled, but no DP-SGD metadata path could be resolved. "
            "Set `target.dpsgd_path`, or place `dpsgd_dic.pkl` inside `target.target_folder`."
        )

    def train(
        self: Self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
        dpsgd_metadata_path: str | None = None,
        virtual_batch_size: int = 16,
    ) -> TrainingOutput:
        """
        Train a DP-SGD compliant model.

        Args:
            dataloader (DataLoader): DataLoader for the training dataset.
            model (torch.nn.Module, optional): The model to be trained.
            criterion (torch.nn.Module, optional): Loss function to optimize.
            optimizer (optim.Optimizer, optional): Optimizer for training.
            epochs (int, optional): Number of training epochs.
            dpsgd_meta_data_path (str): Path to the DP-SGD metadata file containing privacy parameters.
            virtual_batch_size (int): Virtual batch size for DP-SGD training.

        Returns:
            TrainingOutput: Contains the trained model and training metrics, including accuracy and loss history.
        """
        
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training")
        if criterion is None:
            raise ValueError("Criterion must be provided for training")
        if not hasattr(model, "dpsgd"):
            raise ValueError("Model is missing required 'dpsgd' attribute")

        run_dpsgd = bool(model.dpsgd)

        if run_dpsgd:
            dpsgd_metadata_path = self._resolve_dpsgd_metadata_path(dpsgd_metadata_path)
            # Check if the model is compatible with DP-SGD
            errors = ModuleValidator.validate(model, strict=False)
            if len(errors) > 0:
                logger.info("Model has privacy violations. Fixing...")

                # Use ModuleValidator.fix to fix the models privacy violations
                model = ModuleValidator.fix(model)

                # Re-instantiate optimizer with equivalent settings for the fixed model.
                optimizer = _rebuild_optimizer(optimizer, model)
                logger.info(f"Model fixed and {optimizer.__class__.__name__} re-instantiated.")

        # Convert model/optimizer/dataloader to DP-compatible variants when enabled.
        _disable_inplace_activations(model)
        model, optimizer, dataloader, _ = dpsgd(
            model,
            optimizer,
            dataloader,
            dpsgd_path=dpsgd_metadata_path,
        )

        # read hyperparams for training (the parameters for the dataloader are defined in get_dataloader):
        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        device = torch.device("cuda" if cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        accuracy_history = []
        loss_history = []

        if run_dpsgd:
            logger.info("Training with DP-SGD")

            # Use BatchMemoryManager to handle large batch sizes by splitting them into smaller sub-batches
            # Helpful with limited-memory hardware while maintaining the same overall batch size
            with  BatchMemoryManager(
                data_loader=dataloader, 
                max_physical_batch_size=virtual_batch_size, # Set max physical batch size
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
        if hasattr(model, "_module"):
            model = model._module

        results = EvalOutput(
            accuracy=train_accuracy,
            loss=avg_train_loss,
            extra={"accuracy_history": accuracy_history, "loss_history": loss_history},
        )
        return TrainingOutput(model=model, metrics=results)

    def eval(self, loader, model, criterion):
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
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
        def __init__(self, data, targets, transform=None,  indices=None):
            """
            Custom dataset for CelebAHQ data.

            Args:
                data (torch.Tensor): Tensor of input images.
                targets (torch.Tensor): Tensor of labels.
                transform (callable, optional): Optional transform to be applied on the image tensors.
            """
            self.data = data
            self.targets = targets
            self.transform = transform
            self.indices = indices

        def __len__(self):
            """Return the total number of samples."""
            return len(self.targets)

        def __getitem__(self, idx):
            """Retrieve the image and its corresponding label at index 'idx'."""
            image = self.data[idx]
            label = self.targets[idx]

            # Apply transformations to the image if any
            if self.transform:
                image = self.transform(image)

            return image, label
        
        @classmethod
        def from_celebHq(cls, config):
            data_dir = config["data"]["data_dir"]
            train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transform)
            combined_dataset = ConcatDataset([train_dataset, test_dataset])

            # Prepare data loader to iterate over combined_dataset
            loader = DataLoader(combined_dataset, batch_size=1, shuffle=False)

            # Collect all data and targets
            data_list = []
            target_list = []
            for data, target in loader:
                data_list.append(data)  # Remove batch dimension
                target_list.append(target)

            # Concatenate data and targets into large tensors
            data = cat(data_list, dim=0)  # Shape: (N, C, H, W)
            targets = cat(target_list, dim=0)  # Shape: (N,)


            return cls(data, targets)

def _rebuild_optimizer(optimizer: optim.Optimizer, model: torch.nn.Module) -> optim.Optimizer:
    """Recreate optimizer with the same hyperparameters for a new model parameter set."""
    optimizer_class = optimizer.__class__
    optimizer_config = {}
    for group in optimizer.param_groups:
        for key, value in group.items():
            if key != "params":
                optimizer_config[key] = value
    return optimizer_class(model.parameters(), **optimizer_config)


def _disable_inplace_activations(model: torch.nn.Module) -> None:
    """Disable in-place activations that can break Opacus backward hooks."""
    for module in model.modules():
        if hasattr(module, "inplace") and isinstance(module.inplace, bool):
            module.inplace = False


def train_loop(dataloader, model, criterion, optimizer, device, epoch, epochs):

    model.train()
    train_loss, train_acc = 0.0, 0.0
    for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
        labels = labels.long().view(-1)
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = outputs.argmax(dim=1) 
        loss.backward()
        optimizer.step()

        # Accumulate performance of shadow model
        train_acc += pred.eq(labels.view_as(pred)).sum().item()
        train_loss += loss.item() * labels.size(0)

    return train_loss, train_acc

def dpsgd(
        model: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        dataloader: DataLoader = None,
        dpsgd_path: str = "./target_dpsgd/dpsgd_dic.pkl",
    ) -> None:
    """Set the model, optimizer and dataset using DPsgd."""

    if not bool(getattr(model, "dpsgd", False)):
        return model, optimizer, dataloader, None

    logger.info("Training with DP-SGD")

    sample_rate = 1/len(dataloader)
    # Check if the file exists
    if os.path.exists(dpsgd_path):
        # Open and read the pickle file
        with open(dpsgd_path, "rb") as file:
            privacy_engine_dict = pickle.load(file)
        logger.info(f"DP-SGD config loaded from {dpsgd_path}: {privacy_engine_dict}")
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

    if model.dpsgd == False:
        logger.info("DP-SGD flag set to False, Using vanilla training and setting noise_multiplier to 0.0")
        noise_multiplier = 0.0

    # make the model private
    privacy_engine = PrivacyEngine(accountant = "prv")
    priv_model, priv_optimizer, priv_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=privacy_engine_dict["max_grad_norm"],
    )

    return priv_model, priv_optimizer, priv_dataloader, privacy_engine
