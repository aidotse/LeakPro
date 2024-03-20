"""This file contains functions for training and testing the model."""
import time
from ast import Tuple

import torch
from torch import nn
import pickle

def get_optimizer(model: torch.nn.Module, configs: dict):
    optimizer = configs.get("optimizer", "SGD")
    learning_rate = configs.get("learning_rate", 0.001)
    weight_decay = configs.get("weight_decay", 0)
    momentum = configs.get("momentum", 0)
    print(f"Load the optimizer {optimizer}: ", end=" ")
    print(f"Learning rate {learning_rate}", end=" ")
    print(f"Weight decay {weight_decay} ")

    if optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer == "Adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    else:
        raise NotImplementedError(
            f"Optimizer {optimizer} has not been implemented. Please choose from SGD or Adam"
        )

# Test Function
def inference(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str) -> Tuple(float, float):
    """Evaluate the model performance on the test loader
    Args:
        model (torch.nn.Module): Model for evaluation
        loader (torch.utils.data.DataLoader): Data Loader for testing
        device (str): GPU or CPU
    Return:
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
            data, target = data.to(device), target.to(device)
            # Cast target to long tensor
            target = target.long()

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

        # Move model back to CPU
        model.to("cpu")

        # Return loss and accuracy
        return loss, acc

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    configs: dict,
    test_loader: torch.utils.data.DataLoader = None,
    data_split: dict = None,
):
    """Train the model based on on the train loader
    Args:
        model(nn.Module): Model for evaluation.
        train_loader(torch.utils.data.DataLoader): Data loader for training.
        configs (dict): Configurations for training.
    Return:
        nn.Module: Trained model.
    """
    # Get the device for training
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the model to the device
    model.to(device)
    model.train()
    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs)
    # Get the number of epochs for training
    epochs = configs["train"]["epochs"]

    # Loop over each epoch
    for epoch_idx in range(epochs):
        start_time = time.time()
        train_loss, train_acc = 0, 0
        # Loop over the training set
        model.train()
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            # Cast target to long tensor
            target = target.long()

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

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

        print(f"Epoch: {epoch_idx+1}/{epochs} |", end=" ")
        print(f"Train Loss: {train_loss/len(train_loader):.8f} ", end=" ")
        print(f"Train Acc: {float(train_acc)/len(train_loader.dataset):.8f} ", end=" ")

        test_loss, test_acc = inference(model, test_loader, device)

        print(f"Test Loss: {float(test_loss):.8f} ", end=" ")
        print(f"Test Acc: {float(test_acc):.8f} ", end=" ")
        print(f"One step uses {time.time() - start_time:.2f} seconds")

    # Move the model back to the CPU
    model.to("cpu")
    
    save_model_and_metadata(model, data_split, configs, train_acc, test_acc, train_loss, test_loss)

    # Return the model
    return model

def save_model_and_metadata(model: torch.nn.Module, data_split: dict, configs: dict, train_acc: float, test_acc: float, train_loss: float, test_loss: float):
    # Save model and metadata
    model_metadata_dict = {"model_metadata": {}, "current_idx": 0}
    model_idx = model_metadata_dict["current_idx"]
    model_metadata_dict["current_idx"] += 1
    
    log_dir = configs["run"]["log_dir"]
    
    with open(f"{log_dir}/model_{model_idx}.pkl", "wb") as f:
       torch.save(model.state_dict(), f)
    meta_data = {}

    meta_data["train_split"] = data_split["train_indices"]
    meta_data["test_split"] = data_split["test_indices"]
    meta_data["num_train"] = len(data_split["train_indices"])
    meta_data["optimizer"] = configs["train"]["optimizer"]
    meta_data["batch_size"] = configs["train"]["batch_size"]
    meta_data["epochs"] = configs["train"]["epochs"]
    meta_data["model_name"] = configs["train"]["model_name"]
    meta_data["model_idx"] = model_idx
    meta_data["learning_rate"] = configs["train"]["learning_rate"]
    meta_data["weight_decay"] = configs["train"]["weight_decay"]
    meta_data["model_path"] = f"{log_dir}/model_{model_idx}.pkl"
    meta_data["train_acc"] = train_acc
    meta_data["test_acc"] = test_acc
    meta_data["train_loss"] = train_loss
    meta_data["test_loss"] = test_loss
    meta_data["dataset"] = configs["data"]["dataset"]

    model_metadata_dict["model_metadata"][model_idx] = meta_data
    with open(f"{log_dir}/models_metadata.pkl", "wb") as f:
        pickle.dump(model_metadata_dict, f)
    