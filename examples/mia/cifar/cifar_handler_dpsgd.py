"""Module containing the class to handle the user input for the CIFAR10/100 dataset."""

import os
import pickle

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

import torch
from torch import cuda, device, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput

class CifarInputHandlerDPsgd(AbstractInputHandler):
    """Class to handle the user input for the CIFAR100 dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)
        print(configs)

    def get_criterion(self)->None:
        """Set the CrossEntropyLoss for the model."""
        return CrossEntropyLoss()

    def get_optimizer(self, model:torch.nn.Module) -> None:
        """Set the optimizer for the model."""
        learning_rate = 0.1
        momentum = 0.8
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    def dpsgd(
        self,
        model: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        dataloader: DataLoader = None,
    ) -> None:
        """Set the model, optimizer and dataset using DPsgd."""

        print("Training shadow models with DP-SGD")
        dpsgd_path = "./target_dpsgd/dpsgd_dic.pkl"

        sample_rate = 1/len(dataloader)
        # Check if the file exists
        if os.path.exists(dpsgd_path):
            # Open and read the pickle file
            with open(dpsgd_path, "rb") as file:
                privacy_engine_dict = pickle.load(file)
            print("Pickle file loaded successfully!")
            print("Data:", privacy_engine_dict)
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
        self.priv_model, self.priv_optimizer, self.priv_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm= privacy_engine_dict["max_grad_norm"],
        )

        return privacy_engine

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
        dpsgd: bool = True,
    ) -> TrainingOutput:

        """Model training procedure."""
        self.priv_model = model
        self.priv_dataloader = dataloader
        self.priv_optimizer = optimizer

        if dpsgd:
            privacy_engine = self.dpsgd(model, optimizer, dataloader)

        # read hyperparams for training (the parameters for the dataloader are defined in get_dataloader):
        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        self.priv_model.to(gpu_or_cpu)

        # training loop
        for epoch in range(epochs):
            train_loss, train_acc, total_samples = 0, 0, 0
            self.priv_model.train()
            for inputs, labels in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.long()
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
                
                self.priv_model.zero_grad()
                self.priv_optimizer.zero_grad()

                outputs = self.priv_model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs.argmax(dim=1) 
                loss.backward()
                self.priv_optimizer.step()

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.data.view_as(pred)).sum()
                train_loss += loss.item()
                
        avg_train_loss = train_loss / len(dataloader.dataset)
        train_accuracy = train_acc / len(dataloader.dataset) 
        
        self.priv_model.to("cpu")

        output_dict = {"model": self.priv_model, "metrics": {"accuracy": train_accuracy, "loss": avg_train_loss}}
        output = TrainingOutput(**output_dict)
        
        del self.priv_model
        del self.priv_optimizer
        del self.priv_dataloader

        return output
