"""Huang, Yangisibo, et al. "Evaluating Gradient Inversion Attacks and Defenses in Federated Learning."."""

import os
from collections.abc import Generator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
from optuna.trial import Trial
from torch import Tensor, device
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.fl_utils.data_utils import GiaDataModalityExtension, GiaImageExtension
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train
from leakpro.fl_utils.model_utils import InferredBNFeatureHook
from leakpro.fl_utils.similarity_measurements import cosine_similarity_weights, total_variation
from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.import_helper import Callable, Self
from leakpro.utils.logger import logger


@dataclass
class GIABaseRunningConfig:
    """Possible configs for the Gradients attack."""

    # total variation scale for smoothing the reconstructions after each iteration
    tv_reg: float = 0.052
    # learning rate on the attack optimizer
    attack_lr: float = 0.1
    # iterations for the attack steps
    at_iterations: int = 10000
    # MetaOptimizer, see MetaSGD for implementation
    optimizer: object = field(default_factory=lambda: MetaSGD())
    # Client loss function
    criterion: object = field(default_factory=lambda: CrossEntropyLoss(reduction="mean"))
    # Data modality extension
    data_extension: GiaDataModalityExtension = field(default_factory=lambda: GiaImageExtension())
    # Number of epochs for the client attack
    epochs: int = 1
    # if to use median pool 2d on images, can improve attack on high higher resolution (100+)
    median_pooling: bool = False
    # bn regularizer
    bn_reg: float = 0.00016
    # if we compare difference only for top 10 layers with largest changes. Potentially good for larger models.
    top10norms: bool = False


class GIABaseRunning(AbstractGIA):
    """Gradient inversion attack by us."""

    def __init__(self: Self, model: Module, client_loader: DataLoader, data_mean: Tensor, data_std: Tensor,
                train_fn: Optional[Callable] = None, configs: Optional[GIABaseRunningConfig] = None,
                optuna_trial_data: list = None, exp_name: str = "gradient_inversion") -> None:
        super().__init__()
        self.original_model = model
        self.model = deepcopy(self.original_model)
        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None

        self.configs = configs if configs is not None else GIABaseRunningConfig()
        self.optuna_trial_data = optuna_trial_data

        self.client_loader = client_loader
        self.train_fn = train_fn if train_fn is not None else train
        self.data_mean = data_mean
        self.data_std = data_std

        # required for optuna to save the best hyperparameters
        self.attack_cache_folder_path = f"leakpro_output/results/{exp_name}"
        os.makedirs(self.attack_cache_folder_path, exist_ok=True)

        logger.info("Attack initialized.")

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Our attack."
        reference_str = """"""
        summary_str = ""
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:  # noqa: C901
        """Prepare the attack.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """
        gpu_or_cpu = device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(gpu_or_cpu)
        # calculate number of elements per channel to use to un-bias running var.
        bn_channel_element_counts = []
        def bn_capture_n_hook(module: Module, input: torch.tensor, output:torch.tensor) -> None:  # noqa: ARG001
            input_tensor = input[0]
            b, c, h, w = input_tensor.shape
            n = b * h * w
            bn_channel_element_counts.append(n)

        # extract running statistics before client update
        pre_step_running_statistics = []
        hooks_capture_n = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                pre_step_running_statistics.append(
                    (
                        module.running_mean.data.clone().detach(),
                        module.running_var.data.clone().detach()))
                hooks_capture_n.append(module.register_forward_hook(bn_capture_n_hook))

        # calculate running bn statistics and get client gradient
        client_gradient = self.train_fn(self.model, self.client_loader,
                                        self.configs.optimizer, self.configs.criterion, self.configs.epochs)
        for h in hooks_capture_n:
            h.remove()
        post_step_running_statistics = []

        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                post_step_running_statistics.append(
                    (
                        module.running_mean.data.clone().detach(),
                        module.running_var.data.clone().detach()))
        used_statistics = []

        for (rm_pre, rv_pre), (rm_post, rv_post) in zip(pre_step_running_statistics, post_step_running_statistics):
            used_mean = 10 * rm_post - 9 * rm_pre
            used_var = 10 * rv_post - 9 * rv_pre
            used_statistics.append((used_mean, used_var))

        client_statistics = []

        for i in range(len(used_statistics)):
            used_mean, used_var = used_statistics[i]
            n = bn_channel_element_counts[i]
            correction_factor = (n - 1) / n
            client_statistics.append((used_mean, used_var * correction_factor))
        # add feature hook
        self.loss_r_feature_layers = []
        start_idx = 0
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self.loss_r_feature_layers.append(InferredBNFeatureHook(
                    module,
                    client_statistics[start_idx][0],
                    client_statistics[start_idx][1]))
                start_idx += 1

        # Stop updating running statistics
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = 0

        (
            self.client_loader,
            self.original,
            self.reconstruction,
            self.reconstruction_labels,
            self.reconstruction_loader
        ) = self.configs.data_extension.get_at_data(self.client_loader)

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
            to client statistics.

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
            rec_loss.backward()
            self.reconstruction.grad.sign_()
            return rec_loss
        return closure

    def _configure_attack(self: Self, configs: dict) -> None:
        pass

    def suggest_parameters(self: Self, trial: Trial) -> None:
        """Suggest parameters to chose and range for optimization for the Huang attack."""
        total_variation = trial.suggest_float("total_variation", 1e-8, 1e-1, log=True)
        bn_reg = trial.suggest_float("bn_reg", 1e-4, 1e-1, log=True)
        attack_lr = trial.suggest_float("attack_lr", 1e-4, 100.0, log=True)
        median_pooling = trial.suggest_int("median_pooling", 0, 1)
        top10norms = trial.suggest_int("top10norms", 0, 1)
        self.configs.bn_reg = bn_reg
        self.configs.attack_lr = attack_lr
        self.configs.tv_reg = total_variation
        self.configs.median_pooling = median_pooling
        self.configs.top10norms = top10norms
        if self.optuna_trial_data is not None:
            trial_data_idx = trial.suggest_categorical(
                "trial_data",
                list(range(len(self.optuna_trial_data)))
            )
            self.client_loader = self.optuna_trial_data[trial_data_idx]
            logger.info(f"Next experiment on trial data idx: {trial_data_idx}")
        logger.info(f"Chosen parameters:\
                    total_variation: {total_variation} \
                    bn_reg: {bn_reg} \
                    attack_lr: {attack_lr} \
                    median_pooling: {median_pooling} \
                    top10norms: {top10norms}")

    def reset_attack(self: Self, new_config:dict) -> None:  # noqa: ARG002
        """Reset attack to initial state."""
        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None
        self.model = deepcopy(self.original_model)
        super().reset_attack()
        self.prepare_attack()
        logger.info("GIA base attack reset to initial state.")
