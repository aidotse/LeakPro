"""Huang, Yangisibo, et al. "Evaluating Gradient Inversion Attacks and Defenses in Federated Learning."."""

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
from leakpro.fl_utils.data_utils import get_at_images
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train
from leakpro.fl_utils.model_utils import BNFeatureHook
from leakpro.fl_utils.similarity_measurements import cosine_similarity_weights, l2_norm, total_variation
from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.import_helper import Callable, Self
from leakpro.utils.logger import logger


@dataclass
class HuangConfig:
    """Possible configs for the Inverting Gradients attack."""

    # total variation scale for smoothing the reconstructions after each iteration
    tv_reg: float = 0.052
    # learning rate on the attack optimizer
    attack_lr: float = 0.1
    # iterations for the attack steps
    at_iterations: int = 2
    # MetaOptimizer, see MetaSGD for implementation
    optimizer: object = field(default_factory=lambda: MetaSGD())
    # Client loss function
    criterion: object = field(default_factory=lambda: CrossEntropyLoss(reduction="mean"))
    # Number of epochs for the client attack
    epochs: int = 1
    # if to use median pool 2d on images, can improve attack on high higher resolution (100+)
    median_pooling: bool = False
    # bn regularizer
    bn_reg: float = 0.00016
    # l2 scale for discouraging high overall pixel intensity
    l2_scale: float = 0
    # if we compare difference only for top 10 layers with largest changes. Potentially good for larger models.
    top10norms: bool = False


class Huang(AbstractGIA):
    """Gradient inversion attack by Huang et al."""

    def __init__(self: Self, model: Module, client_loader: DataLoader, data_mean: Tensor, data_std: Tensor,
                train_fn: Optional[Callable] = None, configs: Optional[HuangConfig] = None) -> None:
        super().__init__()
        self.original_model = model
        self.model = deepcopy(self.original_model)
        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None

        self.configs = configs if configs is not None else HuangConfig()

        self.client_loader = client_loader
        self.train_fn = train_fn if train_fn is not None else train
        self.data_mean = data_mean
        self.data_std = data_std

        # required for optuna to save the best hyperparameters
        self.attack_folder_path = "leakpro_output/attacks/huang"
        os.makedirs(self.attack_folder_path, exist_ok=True)

        self.prepare_attack()
        logger.info("Evaluating with Huang. et al initialized.")

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Inverting gradients"
        reference_str = """Huang, Yangisibo, et al. "Evaluating Gradient Inversion Attacks and Defenses in \
            Federated Learning. Neurips, 2021."""
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
        # add BN feature hook
        self.loss_r_feature_layers = []

        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self.loss_r_feature_layers.append(BNFeatureHook(module))
        # calculate running bn statistics and get client gradient
        client_gradient = self.train_fn(self.model, self.client_loader,
                                        self.configs.optimizer, self.configs.criterion, self.configs.epochs)

        # Stop updating running statistics
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = 0

        self.reconstruction, self.reconstruction_loader = get_at_images(self.client_loader)
        self.reconstruction.requires_grad = True
        self.client_gradient = [p.detach() for p in client_gradient]


    def run_attack(self:Self) -> Generator[tuple[int, Tensor, GIAResults]]:
        """Run the attack and return the combined metric result.

        Returns
        -------
            GIAResults: Container for results on GIA attacks.

        """
        return self.generic_attack_loop(self.configs, self.gradient_closure, self.configs.at_iterations, self.reconstruction,
                                        self.data_mean, self.data_std, self.configs.attack_lr, self.configs.median_pooling,
                                        self.client_loader, self.reconstruction_loader)


    def gradient_closure(self: Self, optimizer: torch.optim.Optimizer) -> Callable:
        """Returns a closure function that calculates loss and gradients."""
        def closure() -> torch.Tensor:
            """Computes the reconstruction loss and performs backpropagation.

            This function computes the gradient and reconstruction loss using cosine similarity between
            original gradients and gradients from reconstruction images. Total variation, BN update distance
            to running statistics, and L2 norm are added to the loss based on scalars.

            Returns
            -------
                torch.Tensor: The reconstruction loss.

            """
            optimizer.zero_grad()
            self.model.zero_grad()
            gradient = self.train_fn(self.model, self.reconstruction_loader, self.configs.optimizer,
                                     self.configs.criterion, self.configs.epochs)
            rec_loss = cosine_similarity_weights(gradient, self.client_gradient, self.configs.top10norms)

            loss_r_feature = sum([
                mod.r_feature
                for (idx, mod) in enumerate(self.loss_r_feature_layers)
            ])

            # Add the TV loss term to penalize large variations between pixels, encouraging smoother images.
            rec_loss += self.configs.tv_reg * total_variation(self.reconstruction)
            rec_loss += self.configs.bn_reg * loss_r_feature
            rec_loss += self.configs.l2_scale * l2_norm(self.reconstruction)
            rec_loss.backward()
            self.reconstruction.grad.sign_()
            return rec_loss
        return closure

    def _configure_attack(self: Self, configs: dict) -> None:
        pass

    def suggest_parameters(self: Self, trial: Trial) -> None:
        """Suggest parameters to chose and range for optimization for the Huang attack."""
        total_variation = trial.suggest_float("total_variation", 1e-6, 1e-1, log=True)
        bn_reg = trial.suggest_float("bn_reg", 1e-4, 1e-1, log=True)
        self.configs.tv_reg = total_variation
        self.configs.bn_reg = bn_reg

    def reset_attack(self: Self, new_config:dict) -> None:  # noqa: ARG002
        """Reset attack to initial state."""
        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None
        self.model = deepcopy(self.original_model)
        self.prepare_attack()
        logger.info("Huang attack reset to initial state.")
