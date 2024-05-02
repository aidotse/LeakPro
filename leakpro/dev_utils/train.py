"""Contains functions for training and testing the model."""

import logging
import pickle
import time
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from leakpro.import_helper import Tuple


def get_optimizer(model: torch.nn.Module, configs: dict, logger: logging.Logger = None) -> torch.optim.Optimizer:
    """Get the optimizer for training the model.

    Args:
    ----
        model (torch.nn.Module): Model for optimization.
        configs (dict): Configurations for optimization.
        logger (logging.Logger, optional): Logger for logging information (default: None).

    Returns:
    -------
        torch.optim.Optimizer: Optimizer for training the model.

    """
    optimizer = configs['train'].get("optimizer", "SGD")
    learning_rate = configs['train'].get("learning_rate", 0.01)
    weight_decay = configs['train'].get("weight_decay", 0)
    momentum = configs['train'].get("momentum", 0)
    
    logger.info(f"Load the optimizer {optimizer}")
    logger.info(f"Learning rate {learning_rate}")
    logger.info(f"Weight decay {weight_decay} ")

    if optimizer == "SGD":
        logger.info(f"Momentum {momentum} ")
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    if optimizer == "Adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    raise NotImplementedError(
        f"Optimizer {optimizer} has not been implemented. Please choose from SGD or Adam"
    )


# Test Function
def inference(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> Tuple[float, float]:
    """Evaluate the model performance on the test loader.

    Args:
    ----
        model (torch.nn.Module): Model for evaluation
        loader (torch.utils.data.DataLoader): Data Loader for testing
        device (str): GPU or CPU

    Return:
    ------
        loss (float): Loss for the given model on the test dataset.
        acc (float): Accuracy for the given model on the test dataset.

    """
    # Setting model to eval mode and moving to specified device
    model.eval()
    model.to(device)

    # Assigning variables for computing loss and accuracy
    loss, acc, criterion = 0, 0, nn.CrossEntropyLoss()


    # Disable gradient calculation to save memory
    with torch.no_grad():
        for data, target in loader:
            # Moving data and target to the device
            data, target = data.to(device), target.to(device)  # noqa: PLW2901
            # Cast target to long tensor
            target = target.long()  # noqa: PLW2901

            # Computing output and loss
            output = model(data)
            loss += criterion(output, target).item()

            # Computing accuracy
            pred = output.data.max(1, keepdim=True)[1]
            acc += pred.eq(target.data.view_as(pred)).sum()

        # Averaging the losses
        loss /= len(loader)

        # Calculating accuracy
        acc = float(acc) / len(loader.dataset)

        # Return loss and accuracy
        return loss, acc

def train(  # noqa: PLR0913
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    configs: dict,
    test_loader: torch.utils.data.DataLoader = None,
    data_split: dict = None,
    logger: logging.Logger = None,
) -> torch.nn.Module:
    """Train the model based on on the train loader.

    Args:
    ----
        model(nn.Module): Model for evaluation.
        train_loader(torch.utils.data.DataLoader): Data loader for training.
        configs (dict): Configurations for training.
        test_loader (torch.utils.data.DataLoader, optional): Data loader for testing (default: None).
        data_split (dict, optional): Data split for training and testing (default: None).
        logger (logging.Logger, optional): Logger for logging information (default: None).

    Return:
    ------
        nn.Module: Trained model.

    """
    # Get the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the model to the device
    model.to(device)
    model.train()
    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs, logger)
    # Get the number of epochs for training
    epochs = configs["train"]["epochs"]

    # Loop over each epoch
    for epoch_idx in range(epochs):
        start_time = time.time()
        train_loss, train_acc = 0, 0
        # Loop over the training set
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch_idx + 1}/{epochs}") as pbar:
            for data, target in pbar:
                # Cast target to long tensor
                target = target.long()  # noqa: PLW2901

                # Move data to the device
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)  # noqa: PLW2901

                # Set the gradients to zero
                optimizer.zero_grad()

                # Get the model output
                output = model(data)
                # Calculate the loss
                loss = criterion(output, target)
                pred = output.data.max(1, keepdim=True)[1]
                train_acc += pred.eq(target.data.view_as(pred)).sum()
                # Perform the backward pass
                loss.backward()
                # Take a step using optimizer
                optimizer.step()
                # Add the loss to the total loss
                train_loss += loss.item()

        # Log the training loss and accuracy
        log_train_str = f"Epoch: {epoch_idx+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.8f} | Train Acc: {float(train_acc)/len(train_loader.dataset):.8f} | One step uses {time.time() - start_time:.2f} seconds"  # noqa: E501
        logger.info(log_train_str)

        test_loss, test_acc = inference(model, test_loader, device)

        log_test_str = f"Epoch: {epoch_idx+1}/{epochs} | Test Loss: {test_loss:.8f} | Test Acc: {test_acc:.8f}"
        logger.info(log_test_str)

    # Move the model back to the CPU
    model.to("cpu")

    save_model_and_metadata(
        model, data_split, configs, train_acc, test_acc, train_loss, test_loss, optimizer, criterion
    )

    # Return the model
    return model


def save_model_and_metadata(  # noqa: PLR0913
    model: torch.nn.Module,
    data_split: dict,
    configs: dict,
    train_acc: float,
    test_acc: float,
    train_loss: float,
    test_loss: float,
    optimizer: optim.Optimizer,
    loss: nn.Module,
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
        optimizer (str): Optimizer used for training.
        loss (str): Loss function used for training.

    """
    # Save model and metadata
    model_metadata_dict = {"model_metadata": {}}

    log_dir = configs["run"]["log_dir"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/target_model.pkl", "wb") as f:
        torch.save(model.state_dict(), f)
    meta_data = {}

    meta_data["init_params"] = model.init_params if hasattr(model, "init_params") else {}
    meta_data["train_indices"] = data_split["train_indices"]
    meta_data["test_indices"] = data_split["test_indices"]
    meta_data["num_train"] = len(data_split["train_indices"])

    # read out optimizer parameters
    meta_data["optimizer"] = {}
    meta_data["optimizer"]["name"] = optimizer.__class__.__name__.lower()
    meta_data["optimizer"]["lr"] = optimizer.param_groups[0].get("lr", 0)
    meta_data["optimizer"]["weight_decay"] = optimizer.param_groups[0].get("weight_decay", 0)
    meta_data["optimizer"]["momentum"] = optimizer.param_groups[0].get("momentum", 0)
    meta_data["optimizer"]["dampening"] = optimizer.param_groups[0].get("dampening", 0)
    meta_data["optimizer"]["nesterov"] = optimizer.param_groups[0].get("nesterov", False)

    # read out loss parameters
    meta_data["loss"] = {}
    meta_data["loss"]["name"] = loss.__class__.__name__.lower()

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
    with open(f"{log_dir}/model_metadata.pkl", "wb") as f:
        pickle.dump(model_metadata_dict, f)
