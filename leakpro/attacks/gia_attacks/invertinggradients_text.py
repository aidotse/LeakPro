"""Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?."."""
import os
from collections.abc import Generator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
from optuna.trial import Trial
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.fl_utils.data_utils import GiaImageExtension
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train
from leakpro.fl_utils.similarity_measurements import (
    cosine_similarity_weights, 
    total_variation, 
    l2_distance
)

from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.import_helper import Callable, Self
from leakpro.utils.logger import logger
from leakpro.fl_utils.data_utils import get_used_tokens
from leakpro.attacks.gia_attacks.gia_closures import GIAClossure

@dataclass
class InvertingConfig:
    """Possible configs for the Inverting Gradients attack."""

    # total variation scale for smoothing the reconstructions after each iteration
    tv_reg: float = 1.0e-06
    # learning rate on the attack optimizer
    attack_lr: float = 0.1
    # iterations for the attack steps
    at_iterations: int = 8000
    # MetaOptimizer, see MetaSGD for implementation
    optimizer: object = field(default_factory=lambda: MetaSGD())
    # Client loss function
    criterion: object = field(default_factory=lambda: CrossEntropyLoss())
    # Data modality extension
    data_extension: object = field(default_factory=lambda: GiaImageExtension())
    # Number of epochs for the client attack
    epochs: int = 1
    # if to use median pool 2d on images, can improve attack on high higher resolution (100+)
    median_pooling: bool = False
    # if we compare difference only for top 10 layers with largest changes. Potentially good for larger models.
    top10norms: bool = False
    # boolean if using image data or not (temporary solution)
    image_data: bool = True

class InvertingGradients(AbstractGIA):
    """Gradient inversion attack by Geiping et al."""

    def __init__(self: Self, model: Module, client_loader: DataLoader, data_mean: Tensor, data_std: Tensor,
                 train_fn: Optional[Callable] = None, configs: Optional[InvertingConfig] = None, optuna_trial_data: list = None) -> None:
        super().__init__()
        self.original_model = model
        self.model = deepcopy(self.original_model)
        self.client_loader = client_loader
        self.train_fn = train_fn if train_fn is not None else train
        self.data_mean = data_mean
        self.data_std = data_std
        self.configs = configs if configs is not None else InvertingConfig()
        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None
        # required for optuna to save the best hyperparameters
        self.attack_folder_path = "leakpro_output/attacks/inverting_grad"
        os.makedirs(self.attack_folder_path, exist_ok=True)
        # optuna trial data
        self.optuna_trial_data = optuna_trial_data

        logger.info("Inverting gradient initialized.")

        self.prepare_attack()
        print("self.configs: ", self.configs)
        
        
    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Inverting gradients"
        reference_str = """Geiping, Jonas, et al. Inverting gradients-how easy is it to
            break privacy in federated learning? Neurips, 2020."""
        summary_str = ""
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare the attack.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """
        self.model.eval()
        
        client_gradient = self.train_fn(self.model, self.client_loader, self.configs.optimizer,
                                        self.configs.criterion, self.configs.epochs)
        self.client_gradient = [p.detach() for p in client_gradient]
        used_tokens = get_used_tokens(self.model, self.client_gradient)
        print("used_tokens: ", used_tokens)
        print("self.configs.data_extension: ", self.configs.data_extension)
        self.reconstruction, self.reconstruction_loader = self.configs.data_extension.get_at_data(self.client_loader, used_tokens)
                
        if isinstance(self.reconstruction, list):
            for r in self.reconstruction:
                r.requieres_grad = True
        else:
            self.reconstruction.requires_grad = True
        

    def run_attack(self:Self) -> Generator[tuple[int, Tensor, GIAResults]]:
        """Run the attack and return the combined metric result.

        Returns
        -------
            GIAResults: Container for results on GIA attacks.

        """
        
        if self.configs.image_data:
            grad_closure = self.gradient_closure
        else:   
            grad_closure = self.gradient_closure_text
            
        return self.generic_attack_loop(self.configs, grad_closure, self.configs.at_iterations, self.reconstruction,
                                self.data_mean, self.data_std, self.configs.attack_lr, self.configs.median_pooling,
                                self.client_loader, self.reconstruction_loader,self.configs.image_data)


    def gradient_closure(self: Self, optimizer: torch.optim.Optimizer) -> Callable:
        """Returns a closure function that performs a gradient descent step.

        The closure function computes the gradients, calculates the reconstruction loss,
        adds a total variation regularization term, and then performs backpropagation.
        """
        def closure() -> torch.Tensor:
            """Computes the reconstruction loss and performs backpropagation.

            This function zeroes out the gradients of the optimizer and the model,
            computes the gradient and reconstruction loss, logs the reconstruction loss,
            optionally adds a total variation term, performs backpropagation, and optionally
            modifies the gradient of the input image.

            Returns
            -------
                torch.Tensor: The reconstruction loss.

            """
            optimizer.zero_grad()
            self.model.zero_grad()

            gradient = self.train_fn(self.model, self.reconstruction_loader, self.configs.optimizer,
                                     self.configs.criterion, self.configs.epochs)
            rec_loss = cosine_similarity_weights(gradient, self.client_gradient, self.configs.top10norms)
            tv_reg = (self.configs.tv_reg * total_variation(self.reconstruction))
            # Add the TV loss term to penalize large variations between pixels, encouraging smoother images.
            rec_loss += tv_reg
            rec_loss.backward()
            return rec_loss
        return closure
    # TODO: A better solution to handle different gradient closure functions
    def gradient_closure_text(self: Self, optimizer: torch.optim.Optimizer) -> Callable:
        """Returns a closure function that performs a gradient descent step.

        The closure function computes the gradients, calculates the reconstruction loss,
        adds a total variation regularization term, and then performs backpropagation.
        """
        def closure() -> torch.Tensor:
            """Computes the reconstruction loss and performs backpropagation.

            This function zeroes out the gradients of the optimizer and the model,
            computes the gradient and reconstruction loss, logs the reconstruction loss,
            optionally adds a total variation term, performs backpropagation, and optionally
            modifies the gradient of the input image.

            Returns
            -------
                torch.Tensor: The reconstruction loss.

            """
           
            optimizer.zero_grad()
            self.model.zero_grad()
            #print("reconstruction sum inverting: ", self.reconstruction[0].sum())
            gradient = self.train_fn(self.model, self.reconstruction_loader, self.configs.optimizer,
                                     self.configs.criterion, self.configs.epochs)
            
            rec_loss = l2_distance(gradient, self.client_gradient)
            
            # Add Penalties for negative data points
            for r in self.reconstruction:
                rec_loss += 0.01 * torch.sum(torch.relu(-r) ** 2)

            rec_loss.backward()

            #self.reconstruction.grad.sign_()
            return rec_loss
        return closure
    
    def inference_closure(self) -> torch.Tensor:
            """Computes the reconstruction loss and performs backpropagation.

            This function zeroes out the gradients of the optimizer and the model,
            computes the gradient and reconstruction loss, logs the reconstruction loss,
            optionally adds a total variation term, performs backpropagation, and optionally
            modifies the gradient of the input image.

            Returns
            -------
                torch.Tensor: The reconstruction loss.

            """
            #for r in self.reconstruction:
            #    r.requieres_grad = False
            self.model.zero_grad()

            gradient = self.train_fn(self.model, self.reconstruction_loader, self.configs.optimizer,
                                     self.configs.criterion, self.configs.epochs)
            rec_loss = l2_distance(gradient, self.client_gradient)

            # Add Penalties for negative data points
            for r in self.reconstruction:
                rec_loss += 0.01 * torch.sum(torch.relu(-r) ** 2)

            #rec_loss.backward()
            #self.reconstruction.grad.sign_()
            return rec_loss.detach().cpu()

    def _configure_attack(self: Self, configs: dict) -> None:
        pass

    def suggest_parameters(self: Self, trial: Trial) -> None:
        """Suggest parameters to chose and range for optimization for the Inverting Gradient attack."""
        total_variation = trial.suggest_float("total_variation", 1e-8, 1e-1, log=True)
        attack_lr = trial.suggest_float("attack_lr", 1e-4, 100.0, log=True)
        median_pooling = trial.suggest_int("median_pooling", 0, 1)
        top10norms = trial.suggest_int("top10norms", 0, 1)
        self.configs.attack_lr = attack_lr
        self.configs.tv_reg = total_variation
        self.configs.median_pooling = median_pooling
        self.configs.top10norms = top10norms
        if self.optuna_trial_data is not None:
            trial_data_idx = trial.suggest_int("trial_data", 0, len(self.optuna_trial_data) - 1)
            self.client_loader = self.optuna_trial_data[trial_data_idx]
            logger.info(f"Next experiment on trial data idx: {trial_data_idx}")
        logger.info(f"Chosen parameters:\
                    total_variation: {total_variation} \
                    attack_lr: {attack_lr} \
                    median_pooling: {median_pooling} \
                    top10norms: {top10norms}")

    def reset_attack(self: Self, new_config:dict) -> None:  # noqa: ARG002
        """Reset attack to initial state."""
        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None
        self.model = deepcopy(self.original_model)
        self.prepare_attack()
        logger.info("Inverting attack reset to initial state.")

    def get_configs(self: Self) -> dict:
        """Return configs used for attack."""
        return self.configs
    
    
