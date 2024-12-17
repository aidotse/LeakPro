"""Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?."."""
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.fl_utils.data_utils import get_at_images
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.img_utils import MedianPool2d, dataloaders_psnr, total_variation
from leakpro.fl_utils.model_utils import BNFeatureHook
from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.import_helper import Callable, Self
from leakpro.utils.logger import logger


@dataclass
class GradInversionConfig:
    """Possible configs for the Inverting Gradients attack."""

    # total variation scale for smoothing the reconstructions after each iteration
    total_variation: float = 1.0e-06
    # learning rate on the attack optimizer
    attack_lr: float = 0.1
    # iterations for the attack steps
    at_iterations: int = 8000
    # MetaOptimizer, see MetaSGD for implementation
    optimizer: object = field(default_factory=lambda: MetaSGD())
    # Client loss function
    criterion: object = field(default_factory=lambda: CrossEntropyLoss())
    # Number of epochs for the client attack
    epochs: int = 1
    # if to use median pool 2d on images, can improve attack on high higher resolution (100+)
    median_pooling: bool = False
    # if we compare difference only for top 10 layers with largest changes. Potentially good for larger models.
    top10norms: bool = False
    # bn regulizer
    bn_reg: float = 0.001

class GradInversion(AbstractGIA):
    """Gradient inversion attack by Geiping et al."""

    def __init__(self: Self, model: Module, client_loader: DataLoader, train_fn: Callable,
                 data_mean: Tensor, data_std: Tensor, configs: GradInversionConfig) -> None:
        super().__init__()
        self.model = model
        self.client_loader = client_loader
        self.train_fn = train_fn
        self.data_mean = data_mean
        self.data_std = data_std
        self.t_v_scale = configs.total_variation
        self.attack_lr = configs.attack_lr
        self.iterations = configs.at_iterations
        self.optimizer = configs.optimizer
        self.criterion = configs.criterion
        self.epochs = configs.epochs
        self.median_pooling = configs.median_pooling
        self.top10norms = configs.top10norms
        self.bn_reg = configs.bn_reg

        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None
        logger.info("Inverting gradient initialized.")
        self.prepare_attack()

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
        # add feature hook
        self.loss_r_feature_layers = []

        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.reset_running_stats()
                self.loss_r_feature_layers.append(BNFeatureHook(module))

        self.reconstruction, self.reconstruction_loader = get_at_images(self.client_loader)
        self.reconstruction.requires_grad = True
        client_gradient = self.train_fn(self.model, self.client_loader, self.optimizer, self.criterion, self.epochs)
        self.client_gradient = [p.detach() for p in client_gradient]

        # Make feature hook ready for attacks
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = 0
                # self.training = False
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(True)
                if hasattr(module, "bias"):
                    module.bias.requires_grad_(True)

    def run_attack(self:Self) -> GIAResults:
        """Run the attack and return the combined metric result.

        Returns
        -------
            GIAResults: Container for results on GIA attacks.

        """
        optimizer = torch.optim.Adam([self.reconstruction], lr=self.attack_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[self.iterations // 2.667, self.iterations // 1.6,

                                                                        self.iterations // 1.142], gamma=0.1)
        for i in range(self.iterations):
            # loss function which does training and compares distance from reconstruction training to the real training.
            closure = self.gradient_closure(optimizer)

            loss = optimizer.step(closure)
            scheduler.step()
            with torch.no_grad():
                self.reconstruction.data = torch.max(
                    torch.min(self.reconstruction, (1 - self.data_mean) / self.data_std), -self.data_mean / self.data_std
                    )
                if (i +1) % 500 == 0 and self.median_pooling:
                    self.reconstruction.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(self.reconstruction)
            if i % 250 == 0:
                logger.info(f"Iteration {i}, loss {loss}")
            # Chose image who has given least loss
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_reconstruction = deepcopy(self.reconstruction_loader)
                self.best_reconstruction_round = i
                logger.info(f"New best loss: {loss} on round: {i}")

        return GIAResults(self.client_loader, self.best_reconstruction,
                          dataloaders_psnr(self.client_loader, self.reconstruction_loader), self.data_mean, self.data_std)


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

            gradient = self.train_fn(self.model, self.reconstruction_loader, self.optimizer, self.criterion, self.epochs)
            rec_loss = self.reconstruction_costs(gradient, self.client_gradient)

            loss_r_feature = sum([
                mod.r_feature
                for (idx, mod) in enumerate(self.loss_r_feature_layers)
            ])

            rec_loss += self.bn_reg * loss_r_feature
            # Add the TV loss term to penalize large variations between pixels, encouraging smoother images.
            rec_loss += (self.t_v_scale * total_variation(self.reconstruction))
            rec_loss.backward()
            self.reconstruction.grad.sign_()
            return rec_loss
        return closure

    def reconstruction_costs(self: Self, client_gradient: torch.Tensor, reconstruction_gradient: torch.Tensor) -> torch.Tensor:
        """Computes the reconstruction costs between client gradients and the reconstruction gradient.

        This function calculates the pairwise costs between each client gradient and the reconstruction gradient
        using the cosine similarity measure. The costs are accumulated and averaged over all client gradients.

        Returns
        -------
            torch.Tensor: The average reconstruction cost.

        """
        if self.top10norms:
            _, indices = torch.topk(torch.stack([p.norm() for p in reconstruction_gradient], dim=0), 10)
        else:
            indices = torch.arange(len(reconstruction_gradient))
        weights = reconstruction_gradient[0].new_ones(len(reconstruction_gradient))
        total_costs = 0
        pnorm = [0, 0]
        costs = 0
        for i in indices:
            costs -= (client_gradient[i] * reconstruction_gradient[i]).sum() * weights[i]
            pnorm[0] += client_gradient[i].pow(2).sum() * weights[i]
            pnorm[1] += reconstruction_gradient[i].pow(2).sum() * weights[i]
        costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        total_costs += costs
        return total_costs / len(client_gradient)

    def _configure_attack(self: Self, configs: dict) -> None:
        pass
