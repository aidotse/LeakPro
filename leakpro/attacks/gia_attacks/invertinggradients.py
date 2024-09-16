"""Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?."."""
import torch

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.attacks.utils.util_functions import total_variation
from leakpro.metrics.attack_result import GIAResults
from leakpro.user_inputs.abstract_gia_input_handler import AbstractGIAInputHandler
from leakpro.utils.import_helper import Callable, Self
from leakpro.utils.logger import logger


class InvertingGradients(AbstractGIA):
    """Gradient inversion attack by Geiping et al."""

    def __init__(self: Self, handler: AbstractGIAInputHandler, configs: dict) -> None:
        super().__init__(handler)
        self.handler = handler
        self.t_v_scale = configs.get("total_variation")
        self.attack_lr = configs.get("attack_lr")
        self.iterations = configs.get("at_iterations")
        logger.info("Inverting gradient initialized :)")

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
        self.handler.target_model.eval()
        self.client_loader = self.handler.get_client_loader()
        self.data_mean, self.data_std = self.handler.get_meanstd()
        self.reconstruction, self.reconstruction_loader = self.handler.get_at_images()
        self.reconstruction.requires_grad = True

        client_gradient = self.handler.train(self.client_loader, self.handler.get_optimizer())
        self.client_gradient = [p.detach() for p in client_gradient]

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
            if i % 20 == 0:
                logger.info(f"{i}: {loss}")
            # add PSNR calculation and pick best image..
        # Collect client data to one tensor
        return GIAResults(self.client_loader, self.reconstruction_loader, 0, self.data_mean, self.data_std)


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
            self.handler.target_model.zero_grad()

            gradient = self.handler.train(self.reconstruction_loader, self.handler.get_optimizer())
            rec_loss = self.reconstruction_costs(gradient, self.client_gradient)

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
