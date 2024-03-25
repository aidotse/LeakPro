import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Subset

from ..dataset import Dataset
from ..model import Model, PytorchModel


class AttackObjects:
    def __init__(
        self,
        population: Dataset,
        train_test_dataset,
        target_model: Model,
        configs: dict,
    ):
        self._population = population
        self._population_size = len(population)
        self._target_model = PytorchModel(target_model, CrossEntropyLoss())
        self._train_test_dataset = train_test_dataset
        self._num_shadow_models = configs["audit"]["num_shadow_models"]

        self._audit_dataset = {
            # Assuming train_indices and test_indices are arrays of indices, not the actual data
            "data": np.concatenate(
                (
                    train_test_dataset["train_indices"],
                    train_test_dataset["test_indices"],
                )
            ),
            # in_members will be an array from 0 to the number of training indices - 1
            "in_members": np.arange(len(train_test_dataset["train_indices"])),
            # out_members will start after the last training index and go up to the number of test indices - 1
            "out_members": np.arange(
                len(train_test_dataset["train_indices"]),
                len(train_test_dataset["train_indices"])
                + len(train_test_dataset["test_indices"]),
            ),
        }

        # Train shadow models
        if self._num_shadow_models > 0:
            self._shadow_models = []
            self._shadow_train_indices = []

            f_shadow_data = configs["audit"]["f_attack_data_size"]

            for k in range(self._num_shadow_models):
                # Create shadow datasets by sampling from the population
                shadow_data_indices = self.create_shadow_dataset(f_shadow_data)
                shadow_train_loader = DataLoader(Subset(population, shadow_data_indices), batch_size=configs["train"]["batch_size"], shuffle=True,)
                self._shadow_train_indices.append(shadow_data_indices)

                # Initialize a shadow model
                if "adult" in configs["data"]["dataset"]:
                    shadow_model = target_model.__class__(configs["train"]["inputs"], configs["train"]["outputs"])
                elif "cifar10" in configs["data"]["dataset"]:
                    shadow_model = target_model.__class__()

                # Train the shadow model
                shadow_model = self.train_shadow_model(shadow_model, shadow_train_loader, configs = configs)

                # TODO: come up with a way to use different loss functions
                self._shadow_models.append(PytorchModel(shadow_model, CrossEntropyLoss()))



    @property
    def shadow_models(self):
        return self._shadow_models

    @property
    def shadow_train_indices(self):
        return self._shadow_train_indices

    @property
    def population(self):
        return self._population

    @property
    def population_size(self):
        return self._population_size

    @property
    def target_model(self):
        return self._target_model

    @property
    def train_test_dataset(self):
        return self._train_test_dataset

    @property
    def audit_dataset(self):
        return self._audit_dataset

    def create_shadow_dataset(self, f_shadow_data: float, include_in_members:bool=False):

        shadow_data_size = int(f_shadow_data * self.population_size)
        all_index = np.arange(self.population_size)

        # Remove indices corresponding to training data
        if include_in_members is False:
            used_index = self.train_test_dataset["train_indices"]
        else:
            used_index = []

        # pick allowed indices
        selected_index = np.setdiff1d(all_index, used_index, assume_unique=True)
        if shadow_data_size <= len(selected_index):
            selected_index = np.random.choice(selected_index, shadow_data_size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
        return selected_index

    def get_optimizer(self, model: Module, configs: dict):
        optimizer = configs.get("optimizer", "SGD")
        learning_rate = configs.get("learning_rate", 0.01)
        weight_decay = configs.get("weight_decay", 0)
        momentum = configs.get("momentum", 0)
        print(f"Load the optimizer {optimizer}: ", end=" ")
        print(f"Learning rate {learning_rate}", end=" ")
        print(f"Weight decay {weight_decay} ")

        if optimizer == "SGD":
            return SGD(model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        elif optimizer == "Adam":
            return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "AdamW":
            return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        else:
            raise NotImplementedError(
                f"Optimizer {optimizer} has not been implemented. Please choose from SGD or Adam"
            )


    def train_shadow_model(self, shadow_model: Module, shadow_train_loader: DataLoader, shadow_test_loader: DataLoader = None, configs: dict = None):
        """Train the model based on on the train loader
        Args:
            model(nn.Module): Model for evaluation.
            train_loader(torch.utils.data.DataLoader): Data loader for training.
            configs (dict): Configurations for training.

        Return:
        ------
            nn.Module: Trained model.

        """
        # Get the device for training
        device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Set the model to the device
        shadow_model.to(device)
        shadow_model.train()
        # Set the loss function and optimizer
        criterion = CrossEntropyLoss()
        optimizer = self.get_optimizer(shadow_model, configs)
        # Get the number of epochs for training
        epochs = configs["train"]["epochs"]

        # Loop over each epoch
        for epoch_idx in range(epochs):
            start_time = time.time()
            train_loss, train_acc = 0, 0
            # Loop over the training set
            shadow_model.train()
            for data, target in shadow_train_loader:
                # Move data to the device
                data, target = data.to(device, non_blocking=True), target.to(
                    device, non_blocking=True
                )
                # Cast target to long tensor
                target = target.long()

                # Set the gradients to zero
                optimizer.zero_grad(set_to_none=True)

                # Get the model output
                output = shadow_model(data)
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
            print(f"Train Loss: {train_loss/len(shadow_train_loader):.8f} ", end=" ")
            print(f"Train Acc: {float(train_acc)/len(shadow_train_loader.dataset):.8f} ", end=" ")

            #test_loss, test_acc = inference(shadow_model, shadow_test_loader, device)

           #  print(f"Test Loss: {float(test_loss):.8f} ", end=" ")
           # print(f"Test Acc: {float(test_acc):.8f} ", end=" ")
            print(f"One step uses {time.time() - start_time:.2f} seconds")

        # Move the model back to the CPU
        shadow_model.to("cpu")

        # save_model_and_metadata(shadow_model, configs, train_acc, test_acc, train_loss, test_loss)

        # Return the model
        return shadow_model
