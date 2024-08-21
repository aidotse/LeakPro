"""Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?."."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.import_helper import Callable, Self
from leakpro.metrics.attack_result import GIAResults
from leakpro.user_inputs.abstract_gia_input_handler import AbstractGIAInputHandler


class InvertingGradients(AbstractGIA):
    """Gradient inversion attack by Geiping et al."""

    def __init__(self: Self, handler: AbstractGIAInputHandler, configs: dict) -> None:
        super().__init__(handler)
        self.handler = handler
        self.logger.info("Inverting gradient initialized :)")
        self.t_v_scale = configs.get("total_variation")
        self.attack_lr = configs.get("attack_lr")
        self.iterations = configs.get("at_iterations")

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Inverting gradients"
        reference_str = "Geiping, Jonas, et al. Inverting gradients-how easy is it to break privacy in federated learning?(2020)."
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
        self.client_loader = self.handler.client_data
        self.data_mean, self.data_std = self.handler.get_meanstd()
        self.reconstruction = self.handler.init_at_image()

        # Extract the labels from self.handler.client_data
        labels = []
        for _, label in self.handler.client_data:
            labels.extend(label.numpy())
        labels = torch.tensor(labels)
        reconstruction_dataset = TensorDataset(self.reconstruction, labels)
        self.reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)

        client_gradient = self.handler.train(self.client_loader, self.handler.get_optimizer())
        self.client_gradient = [p.detach() for p in client_gradient]

    def run_attack(self:Self) -> GIAResults:
        """Run the attack and return the combined metric result.

        Returns
        -------
            GIAResults: Container for results on GIA attacks.

        """
        self.reconstruction.requires_grad = True
        optimizer = torch.optim.Adam([self.reconstruction], lr=self.attack_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[24000 // 2.667, 24000 // 1.6,

                                                                        24000 // 1.142], gamma=0.1)

        for i in range(self.iterations):
            closure = self.gradient_closure(optimizer)

            loss = optimizer.step(closure)
            scheduler.step()
            with torch.no_grad():
                self.reconstruction.data = torch.max(
                    torch.min(self.reconstruction, (1 - self.data_mean) / self.data_std), -self.data_mean / self.data_std
                    )
            if i % 20 == 0:
                self.logger.info(f"{i}: {loss}")
            # add PSNR calculation and pick best image..
        # Collect client data to one tensor
        all_data = []

        for i in range(len(self.client_loader)):
            x, _ = self.client_loader[i]
            all_data.append(x)

        # Convert lists to tensors
        original_data_tensor = torch.stack(all_data)
        return GIAResults(all_data[0], self.reconstruction, 0, self.data_mean, self.data_std)

    def total_variation(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Anisotropic TV."""
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy


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
            rec_loss = self.reconstruction_costs([gradient], self.client_gradient)

            self.logger.info(f"rec loss {rec_loss}")
            rec_loss += (self.t_v_scale * self.total_variation(self.reconstruction))
            rec_loss.backward()
            self.reconstruction.grad.sign_()
            return rec_loss
        return closure

    def reconstruction_costs(self: Self, client_gradients: torch.Tensor, reconstruction_gradient: torch.Tensor) -> torch.Tensor:
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
        for trial_gradient in client_gradients:
            pnorm = [0, 0]
            costs = 0
            for i in indices:
                costs -= (trial_gradient[i] * reconstruction_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += reconstruction_gradient[i].pow(2).sum() * weights[i]
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

            # Accumulate final costs
            total_costs += costs
        return total_costs / len(client_gradients)

    def _configure_attack(self: Self, configs: dict) -> None:
        pass
