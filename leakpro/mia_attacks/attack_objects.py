"""Module for the AttackObjects class."""

import logging
import os
import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss, Module, functional
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Subset

from leakpro.dataset import Dataset
from leakpro.import_helper import List, Self
from leakpro.model import Model, PytorchModel


class AttackObjects:
    """Class representing the attack objects for the MIA attacks."""

    def __init__(   # noqa: PLR0915, PLR0912, PLR0913, C901
        self:Self,
        population: Dataset,
        train_test_dataset: dict,
        target_model: Model,
        configs: dict,
        logger: logging.Logger = None
    ) -> None:
        """Initialize the AttackObjects class.

        Parameters
        ----------
        population : Dataset
            The population.
        train_test_dataset : dict
            The train test dataset.
        target_model : Model
            The target model.
        configs : dict
            The configurations.
        logger : logging.Logger, optional
            The logger, by default None.

        """
        self._population = population
        self._population_size = len(population)
        self._target_model = PytorchModel(target_model, CrossEntropyLoss())
        self._train_test_dataset = train_test_dataset
        self._num_shadow_models = configs["audit"]["num_shadow_models"]
        self.num_distillation_models_target = configs["loss_traj"]["number_of_traj"]
        self.num_distillation_models_shadow = configs["loss_traj"]["number_of_traj"]
        self.configs = configs
        self.logger = logger
        self._distillation_models_shadow = []
        self._distillation_models_target = []

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

        self.log_dir = configs["run"]["log_dir"]

        path_shadow_models = f"{self.log_dir}/shadow_models"


        # Check if the folder does not exist
        if not os.path.exists(path_shadow_models):
            # Create the folder
            os.makedirs(path_shadow_models)

        # List all entries in the directory
        entries = os.listdir(path_shadow_models)
        number_of_files_to_reuse = len(entries)


        # Train shadow models
        self._shadow_models = []
        shadow_train_data_indices, shadow_test_data_indices, distillation_train_data_indices,  distillation_test_data_indices = self.create_aux_dataset(include_in_members=False)  # noqa: E501
        if self._num_shadow_models > 0:
            self._shadow_train_indices = []
            self._shadow_test_indices = []

            for k in range(self._num_shadow_models):
                if "adult" in configs["data"]["dataset"]:
                    shadow_model = target_model.__class__(configs["train"]["inputs"], configs["train"]["outputs"])
                elif "cifar10" in configs["data"]["dataset"]:
                    shadow_model = target_model.__class__()
                elif "cinic10" in configs["data"]["dataset"]:
                    shadow_model = target_model.__class__(configs)

                if number_of_files_to_reuse > 0:
                    self.logger.info("Loading trained shadow model")
                    shadow_model.load_state_dict(torch.load(f"{path_shadow_models}/model_{k}.pkl"))
                    self._shadow_models.append(PytorchModel(shadow_model, CrossEntropyLoss()))
                    #TODO: load data indices
                    number_of_files_to_reuse -= 1
                else:
                    # Create a dataloder for training the shadow model
                    self._shadow_train_indices = shadow_train_data_indices
                    shadow_train_loader = DataLoader(Subset(population, shadow_train_data_indices),
                                                    batch_size=configs["train"]["batch_size"],
                                                    shuffle=True,)

                    # Create a dataloder for testing the shadow model
                    self._shadow_test_indices = shadow_test_data_indices
                    self._shadow_train_indices = shadow_train_data_indices
                    shadow_test_loader = DataLoader(Subset(population, shadow_test_data_indices),
                                                    batch_size=configs["train"]["batch_size"],
                                                    shuffle=True,)
                    # Train the shadow model
                    shadow_model = self.train_test_shadow_model(shadow_model, shadow_train_loader,
                                                                 shadow_test_loader,  configs = configs)

                    #save the shadow model
                    torch.save(shadow_model.state_dict(),
                            f"{path_shadow_models}/model_{k}.pkl")

                    # TODO: come up with a way to use different loss functions
                    #TODO : save the meta data and indices
                    self._shadow_models.append(PytorchModel(shadow_model, CrossEntropyLoss()))


        # Train knowledge distillation models of the target model
        self._distillation_train_indices = []
        self._distillation_test_indices = []
        if self.num_distillation_models_target > 0:
            self._distillation_models_target = []

            # Distillation models of target
            path_distillation_models_target = f"{self.log_dir}/distillation_models_target"

            # Check if the folder does notexist
            if not os.path.exists(path_distillation_models_target):
                #Create the folder
                os.makedirs(path_distillation_models_target)
            # List all entries in the directory
            number_of_reuse_distillation_target_models = len(os.listdir(path_distillation_models_target))
            if number_of_reuse_distillation_target_models > 0:
                if number_of_reuse_distillation_target_models != self.num_distillation_models_target:
                    # TODO: add how to handle this situation
                    raise ValueError(
                    f" number of reused distillation models ({number_of_reuse_distillation_target_models}) "
                    f"and requested number of distillation model in config file "
                    f"({self.num_distillation_models_target}) are not equal")

                self.logger.info(f"Loading {number_of_reuse_distillation_target_models} trained distillated models of the target")
                for k in range(number_of_reuse_distillation_target_models):
                    if "adult" in configs["data"]["dataset"]:
                        distillation_target_model = target_model.__class__(configs["train"]["inputs"],
                                                                           configs["train"]["outputs"])
                    elif "cifar10" in configs["data"]["dataset"]:
                        distillation_target_model = target_model.__class__()
                    elif "cinic10" in configs["data"]["dataset"]:
                        distillation_target_model = target_model.__class__(configs)
                    distillation_target_model.load_state_dict(torch.load(f"{path_distillation_models_target}/model_{k}.pkl"))
                    # TODO: load data indices
                    self._distillation_models_target.append(PytorchModel(distillation_target_model, CrossEntropyLoss()))
            else:
                self._distillation_train_indices = distillation_train_data_indices
                distillation_train_loader = DataLoader(Subset(population, distillation_train_data_indices),
                                                    batch_size=configs["train"]["batch_size"],
                                                    shuffle=True,)

                # Train the distillation model
                self.train_distillation_models_target( target_model,
                                                self.num_distillation_models_target,
                                                distillation_train_loader , configs=configs)



        # Train knowledge distillation models of the shadow model
        #TODO : ATM the code is for one shadow model
        if self.num_distillation_models_shadow > 0:

            # Distillation models of shadow
            self._distillation_models_shadow = []
            path_distillation_models_shadow = f"{self.log_dir}/distillation_models_shadow"

            # Check if the folder does notexist
            if not os.path.exists(path_distillation_models_shadow):
                #Create the folder
                os.makedirs(path_distillation_models_shadow)
            # List all entries in the directory
            number_of_reuse_distillation_models_shadow = len(os.listdir(path_distillation_models_shadow))
            if number_of_reuse_distillation_models_shadow > 0:
                if number_of_reuse_distillation_models_shadow != self.num_distillation_models_shadow:
                    raise ValueError(
                    f" number of reused distillation models ({number_of_reuse_distillation_models_shadow}) "
                    f"and requested number of distillation model in config file "
                    f"({self.num_distillation_models_shadow}) are not equal")

                self.logger.info(f"Loading {number_of_reuse_distillation_models_shadow} trained distillated models of the shadow")
                for k in range(number_of_reuse_distillation_models_shadow):
                    if "adult" in configs["data"]["dataset"]:
                        distillation_shadow_model = target_model.__class__(configs["train"]["inputs"],
                                                                           configs["train"]["outputs"])
                    elif "cifar10" in configs["data"]["dataset"]:
                        distillation_shadow_model = target_model.__class__()
                    elif "cinic10" in configs["data"]["dataset"]:
                        distillation_shadow_model = target_model.__class__(configs)

                    distillation_shadow_model.load_state_dict(torch.load(f"{path_distillation_models_shadow}/model_{k}.pkl"))
                    self._distillation_models_shadow.append(PytorchModel(distillation_shadow_model,
                                                                         CrossEntropyLoss()))
            else:
                self._distillation_train_indices = distillation_train_data_indices
                distillation_train_loader = DataLoader(Subset(population, distillation_train_data_indices),
                                                    batch_size=configs["train"]["batch_size"],
                                                    shuffle=True,)

                # Train the distillation model
                self.train_distillation_models_shadow( \
                        target_model,self.num_distillation_models_shadow,
                        distillation_train_loader, configs=configs)



    @property
    def shadow_models(self: Self) -> List[Model]:
        """Return the shadow models.

        Returns
        -------
        List[Model]: The shadow models.

        """
        return self._shadow_models

    @property
    def distillation_models_target(self: Self) -> List[Model]:
        """Return the distillation of the target model.

        Returns
        -------
        List[Model]: The disitllation of the target model.

        """
        return self._distillation_models_target

    @property
    def distillation_models_shadow(self: Self) -> List[Model]:
        """Return the distillation of the shadow model.

        Returns
        -------
        List[Model]: The distillation of the shadow model.

        """
        return self._distillation_models_shadow

    @property
    def distillation_target_train_indices(self:Self) -> List[int]:
        """Return the indices of the distillation training data.

        Returns
        -------
        List[int]: The indices of the distillation training data.

        """
        return self._distillation_target_train_indices

    @property
    def distillation_shadow_train_indices(self:Self) -> List[int]:
        """Return the indices of the distillation training data.

        Returns
        -------
        List[int]: The indices of the distillation training data.

        """
        return self._distillation_shadow_train_indices


    @property
    def population(self:Self) -> Dataset:
        """Return the population.

        Returns
        -------
        Dataset: The population.

        """
        return self._population

    @property
    def population_size(self:Self) -> int:
        """Return the size of the population.

        Returns
        -------
        int: The size of the population.

        """
        return self._population_size

    @property
    def target_model(self:Self) -> Model:
        """Return the target model.

        Returns
        -------
        Model: The target model.

        """
        return self._target_model

    @property
    def logger(self:Self) -> logging.Logger:
        """Return the logger.

        Returns
        -------
        Model: The logger object.

        """
        return self._logger

    @property
    def train_test_dataset(self:Self) -> dict:
        """Return the train test dataset.

        Returns
        -------
            dict: The train test dataset.

        """
        return self._train_test_dataset

    @property
    def audit_dataset(self:Self) -> dict:
        """Return the audit dataset.

        Returns
        -------
            dict: The audit dataset.

        """
        return self._audit_dataset

    def create_shadow_dataset(self:Self, f_shadow_data: float, include_in_members:bool=False) -> np.ndarray:
        """Create a shadow dataset by sampling from the population.

        Args:
        ----
            f_shadow_data (float): Fraction of shadow data to be sampled.
            include_in_members (bool, optional): Include in-members in the shadow dataset. Defaults to False.

        Returns:
        -------
            np.ndarray: Array of indices representing the shadow dataset.

        """
        shadow_data_size = int(f_shadow_data * self.population_size)
        all_index = np.arange(self.population_size)

        # Remove indices corresponding to training data
        used_index = self.train_test_dataset["train_indices"] if include_in_members is False else []

        # pick allowed indices
        selected_index = np.setdiff1d(all_index, used_index, assume_unique=True)
        if shadow_data_size <= len(selected_index):
            selected_index = np.random.choice(selected_index, shadow_data_size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
        return selected_index


    def create_aux_dataset(self:Self, include_in_members:bool=False)-> tuple:
        """Create a shadow dataset (train and test) by sampling from the population.

        Args:
        ----
            f_shadow_data (float): Fraction of shadow data to be sampled.
            include_in_members (bool, optional): Include in-members in the shadow dataset. Defaults to False.

        Returns:
        -------
            np.ndarray: Array of indices representing the shadow dataset.

        """
        aux_data_s = self.configs["loss_traj"]["aux_data_size"]
        shadow_train_s = int(self.configs["loss_traj"]["train_shadow_data_size"])
        shadow_test_s = int(self.configs["loss_traj"]["test_shadow_data_size"])
        distill_test_data_s = int(self.configs["loss_traj"]["train_distillation_data_size"]/2)
        distill_train_data_s = int(self.configs["loss_traj"]["train_distillation_data_size"]/2)
        assert aux_data_s == shadow_train_s + shadow_test_s + distill_test_data_s + distill_train_data_s  # noqa: S101

        all_index = np.arange(self.population_size)

        # Remove indices corresponding to training data of the target model
        used_index_train_target = self.train_test_dataset["train_indices"] if include_in_members is False else []

        # Remove indices corresponding to test data of the target model
        used_index_test_target = self.train_test_dataset["test_indices"] if include_in_members is False else []

        # Concatenate the arrays of indices to be removed
        used_indices_target = np.concatenate((used_index_train_target, used_index_test_target))


        # pick allowed indices
        mask = np.isin(all_index, used_indices_target, invert=True)
        unused_indices = all_index[mask]
        if aux_data_s <= len(unused_indices):
            np.random.shuffle(unused_indices)
            shadow_train_indices = unused_indices[:shadow_train_s]
            shadow_test_indices = unused_indices[shadow_train_s:
                                                 shadow_train_s+ shadow_test_s]
            distill_test_data_indices = unused_indices[shadow_train_s+ shadow_test_s:
                                                       shadow_train_s+ shadow_test_s+ distill_test_data_s]
            distill_train_data_indices = unused_indices[ shadow_train_s+ shadow_train_s+
                                                        distill_test_data_s:]

        else:
            raise ValueError("Not enough remaining data points.")

        return shadow_train_indices, shadow_test_indices, distill_test_data_indices,distill_train_data_indices


    def get_optimizer(self:Self, model: Module, configs: dict) -> torch.optim.Optimizer:
        """Get the optimizer for training the model.

        Args:
        ----
            model (nn.Module): Model for training.
            configs (dict): Configurations for training.

        Returns:
        -------
            Optimizer: The optimizer for training the model.

        """
        optimizer = configs.get("optimizer", "SGD")
        learning_rate = configs.get("learning_rate", 0.01)
        weight_decay = configs.get("weight_decay", 0)
        momentum = configs.get("momentum", 0)
        self.logger.info(f"Load the optimizer {optimizer}")
        self.logger.info(f"Learning rate {learning_rate}")
        self.logger.info(f"Weight decay {weight_decay}")

        if optimizer == "SGD":
            return SGD(model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        if optimizer == "Adam":
            return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if optimizer == "AdamW":
            return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        raise NotImplementedError(
            f"Optimizer {optimizer} has not been implemented. Please choose from SGD or Adam"
        )


    def train_test_shadow_model(self:Self, shadow_model: Module, shadow_train_loader: DataLoader,
                               shadow_test_loader: DataLoader, configs: dict = None) -> Module:
        """Train the model based on on the train loader and test loader.

        Args:
        ----
            shadow_model(nn.Module): Model for evaluation.
            shadow_train_loader(torch.utils.data.DataLoader): Data loader for training.
            shadow_test_loader(torch.utils.data.DataLoader): Data loader for testing.
            configs (dict): Configurations for training.

        Return:
        ------
            nn.Module: Trained model.

        """
        self.logger.info(" ************* Training the shado model ************")
        device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Set the model to the device
        shadow_model.to(device)
        # Set the loss function and optimizer
        criterion = CrossEntropyLoss()
        optimizer = self.get_optimizer(shadow_model, configs)
        # Get the number of epochs for training
        epochs = configs["train"]["epochs"]


        # Loop over each epoch
        for epoch_idx in range(epochs):
            start_time = time.time()
            train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0

            # Training Phase
            shadow_model.train()
            for data, target in shadow_train_loader:
                data, target = data.to(device), target.to(device)  # noqa: PLW2901
                target = target.long()  # noqa: PLW2901

                optimizer.zero_grad()
                output = shadow_model(data)
                loss = criterion(output, target)
                pred = output.data.max(1, keepdim=True)[1]
                train_acc += pred.eq(target.data.view_as(pred)).sum()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(shadow_train_loader)
            train_acc = float(train_acc) / len(shadow_train_loader.dataset)

            # Testing Phase
            shadow_model.eval()
            with torch.no_grad():
                for data, target in shadow_test_loader:
                    data, target = data.to(device), target.to(device)  # noqa: PLW2901
                    target = target.long()  # noqa: PLW2901

                    output = shadow_model(data)
                    loss = criterion(output, target)
                    pred = output.data.max(1, keepdim=True)[1]
                    test_acc += pred.eq(target.data.view_as(pred)).sum()
                    test_loss += loss.item()

            test_loss /= len(shadow_test_loader)
            test_acc = float(test_acc) / len(shadow_test_loader.dataset)

            # Logging for both train and test phases
            log_str = f"Epoch: {epoch_idx+1}/{epochs} | " \
                    f"Train Loss: {train_loss:.8f} | Train Acc: {train_acc:.8f} | " \
                    f"Test Loss: {test_loss:.8f} | Test Acc: {test_acc:.8f} | " \
                    f"Epoch Time: {time.time() - start_time:.2f} seconds"
            self.logger.info(log_str)

        # Move the model back to the CPU
        shadow_model.to("cpu")

        return shadow_model


    def train_distillation_models_target(self:Self, target_model: Module,
                                         number_of_distillation_models: int,
                                         distillation_train_loader: DataLoader,
                                         configs: dict = None) -> Module:
        """Train the model based on the train loader.

        Args:
        ----
            target_model (nn.Module): The target model to be trained.
            number_of_distillation_models (int): The number of distillation models to train.
            distillation_train_loader (torch.utils.data.DataLoader): The data loader for training.
            configs (dict, optional): Configurations for training.

        Returns:
        -------
            nn.Module: The trained model.

        """
        path_distillation_models_target = f"{self.log_dir}/distillation_models_target"

        # Get the device for training
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        target_model.to(device)


        # Distillation model
        if "adult" in configs["data"]["dataset"]:
            distillation_target_model = target_model.__class__(configs["train"]["inputs"], configs["train"]["outputs"])
        elif "cifar10" in configs["data"]["dataset"]:
            distillation_target_model = target_model.__class__()
        elif "cinic10" in configs["data"]["dataset"]:
            distillation_target_model = target_model.__class__(configs)

        for d in range(number_of_distillation_models):
            distillation_target_model.to(device)
            target_model.to(device)

            distillation_target_model.train()
            target_model.eval()

            optimizer = self.get_optimizer(distillation_target_model, configs)
            train_loss = 0

            # Loop over each epoch
            self.logger.info(f" *** Training distillation model of target model, epoch: {d}")

            # Loop over the training set
            for data, target_labels in distillation_train_loader:
                # Move data to the device
                data, target_labels = data.to(device, non_blocking=True), target_labels.to(device, non_blocking=True)  # noqa: PLW2901
                target_labels = target_labels.long()  # noqa: PLW2901

                # Output of the distillation model
                output = distillation_target_model(data)
                output_teacher = target_model(data)

                soft_target = torch.zeros_like(output)
                soft_target[torch.arange(soft_target.shape[0]), target_labels] = 1

                loss = KLDivLoss(reduction="batchmean")(functional.log_softmax(output, dim=1),
                                                            functional.softmax(output_teacher,
                                                            dim=1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            torch.save(distillation_target_model.state_dict(), f"{path_distillation_models_target}/epoch_{d}.pkl")

        # Return the model
        return


    def train_distillation_models_shadow(self:Self, target_model: Module,
                                         number_of_distillation_models:int,
                                         distillation_train_loader: DataLoader,
                                         configs: dict = None) -> Module:
        """Train the model based on the train loader.

        Args:
        ----
            target_model (nn.Module): The target model to be distilled.
            number_of_distillation_models (int): The number of distillation models to train.
            distillation_train_loader (torch.utils.data.DataLoader): The data loader for training.
            configs (dict, optional): Configurations for training.

        Returns:
        -------
            nn.Module: The trained distillation model.

        """
        path_distillation_models_shadow = f"{self.log_dir}/distillation_models_shadow"

        # Get the device for training
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        target_model.to(device)


        # Distillation model
        if "adult" in configs["data"]["dataset"]:
            distillation_shadow_model = target_model.__class__(configs["train"]["inputs"], configs["train"]["outputs"])
        elif "cifar10" in configs["data"]["dataset"]:
            distillation_shadow_model = target_model.__class__()
        elif "cinic10" in configs["data"]["dataset"]:
            distillation_shadow_model = target_model.__class__(configs)

        for d in range(number_of_distillation_models):

            distillation_shadow_model.to(device)
            target_model.to(device)

            distillation_shadow_model.train()
            target_model.eval()

            optimizer = self.get_optimizer(distillation_shadow_model, configs)
            train_loss = 0

            # Loop over each epoch
            self.logger.info(f" *** Training distillation model of shadow model, epoch: {d}")

            # Loop over the training set
            for data, target_labels in distillation_train_loader:

                # Move data to the device
                data, target_labels = data.to(device, non_blocking=True), target_labels.to(device, non_blocking=True)  # noqa: PLW2901
                target_labels = target_labels.long()  # noqa: PLW2901

                # Output of the distillation model
                output = distillation_shadow_model(data)
                output_teacher = target_model(data)

                soft_target = torch.zeros_like(output)
                soft_target[torch.arange(soft_target.shape[0]), target_labels] = 1

                loss = KLDivLoss(reduction="batchmean")(functional.log_softmax(output, dim=1),
                                                            functional.softmax(output_teacher.float(),
                                                            dim=1))


                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            torch.save(distillation_shadow_model.state_dict(), f"{path_distillation_models_shadow}/epoch_{d}.pkl")

        # Return the model
        return
