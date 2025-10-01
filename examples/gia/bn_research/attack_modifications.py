import torch
from leakpro.fl_utils.similarity_measurements import cosine_similarity_weights, l2_norm, total_variation
from leakpro.utils.import_helper import Callable, Self
from torch import cuda, device
import time

import torch
from torch import nn
from torch.autograd import Function

class CustomBatchNorm2dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, eps, momentum, training):
        B, C, H, W = input.shape

        if training:
            # Compute batch mean and variance
            mean = input.mean(dim=(0, 2, 3), keepdim=True)  # shape (1,C,1,1)
            var = input.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            # Update running stats
            with torch.no_grad():
                running_mean.mul_(1 - momentum).add_(momentum * mean.view(-1))
                running_var.mul_(1 - momentum).add_(momentum * var.view(-1))
        else:
            # Use running statistics
            mean = running_mean.view(1, C, 1, 1)
            var = running_var.view(1, C, 1, 1)

        # Normalize
        inv_std = torch.rsqrt(var + eps)
        x_hat = (input - mean) * inv_std
        output = weight.view(1, C, 1, 1) * x_hat + bias.view(1, C, 1, 1)

        # Save for backward
        ctx.save_for_backward(x_hat, inv_std, weight)
        ctx.training = training

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, inv_std, weight = ctx.saved_tensors
        B, C, H, W = grad_output.shape

        grad_input = grad_weight = grad_bias = grad_running_mean = grad_running_var = grad_eps = grad_momentum = grad_training = None

        if ctx.training:
            # Gradients for weight and bias
            N = B * H * W
            grad_weight = (grad_output * x_hat).sum(dim=(0, 2, 3))
            grad_bias = grad_output.sum(dim=(0, 2, 3))
            dx_hat = grad_output * weight.view(1, C, 1, 1)

            # Gradient w.r.t. input

            dx_hat_sum = dx_hat.sum(dim=(0, 2, 3), keepdim=True)
            x_hat_dx_hat_sum = (dx_hat * x_hat).sum(dim=(0, 2, 3), keepdim=True)

            grad_input = (dx_hat - x_hat * x_hat_dx_hat_sum / N - dx_hat_sum / N) * inv_std
            # paths back to X:
            # grad_weight (x_hat is normalized X)
            # grad input (through x_hat * x_hat_dx_hat_sum which gives X^2, non-linear?). And dx_hat_sum again. and inv_std)

        else:
            # Eval mode: only scale by weight * inv_std
            grad_weight = (grad_output * x_hat).sum(dim=(0, 2, 3))
            grad_bias = grad_output.sum(dim=(0, 2, 3))
            dx_hat = grad_output * weight.view(1, C, 1, 1)
            grad_input = dx_hat * inv_std
            # paths back to X:
            # grad_weight (x_hat is normalized X)
            # grad input (through inv_std)
            # both linear here?

        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        return CustomBatchNorm2dFunction.apply(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
            self.training
        )


# alteration of prepare attack where we infer the normalization through the update running statistics
def prepare_attack(self:Self) -> None:
    """Prepare the attack.

    This alteration of prepare attack assumes fedSGD where client performs one step and calculates the exact statistics used 
    by the client based on the difference in running statistics. It then places those exact statistics as the running 
    statistics and puts the model in evaluation mode, thus enforcing the model will use the same statistics as the client did.

    Args:
    ----
        self (Self): The instance of the class.

    Returns:
    -------
        None

    """
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    self.model.to(gpu_or_cpu)
    self.model.train()
    bn_channel_element_counts = []
    def bn_capture_n_hook(module, input, output):
        input_tensor = input[0]
        B, C, H, W = input_tensor.shape
        n = B * H * W
        bn_channel_element_counts.append(n)
        # save input for backward in the future

    def bn_capture_save_input_hook(module, input, output):
        module._saved_input_for_hook = input[0].detach()

    def correct_gradient(module, grad_input, grad_output):
        if isinstance(module, torch.nn.BatchNorm2d) and not module.training:
            with torch.no_grad():
                x = module._saved_input_for_hook
                B, C, H, W = x.shape
                N = B * H * W

                weight = module.weight.view(1, C, 1, 1)
                eps = module.eps

                # Use running statistics for eval mode
                running_mean = module.running_mean.view(1, C, 1, 1)
                running_var = module.running_var.view(1, C, 1, 1)
                inv_std = 1.0 / torch.sqrt(running_var + eps)
                x_hat = (x - running_mean) * inv_std

                # Compute gradients w.r.t. input following BN backward
                dx_hat = grad_output[0] * weight
                dx_hat_sum = dx_hat.sum(dim=(0, 2, 3), keepdim=True)
                x_hat_dx_hat_sum = (dx_hat * x_hat).sum(dim=(0, 2, 3), keepdim=True)

                corrected_grad_input = (dx_hat - x_hat * x_hat_dx_hat_sum / N - dx_hat_sum / N) * inv_std

            grad_input = (corrected_grad_input,) + grad_input[1:]
        return grad_input


    pre_step_running_statistics = []
    hooks_capture_n = []
    hook2_capture = []
    hook3_capture = []
    i = 0
    for module in self.model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            pre_step_running_statistics.append((module.running_mean.data.clone().detach(), module.running_var.data.clone().detach()))
            hooks_capture_n.append(module.register_forward_hook(bn_capture_n_hook))
            hook2_capture.append(module.register_backward_hook(correct_gradient))
            hook3_capture.append(module.register_forward_hook(bn_capture_save_input_hook))
        i+=1
    # calculate running bn statistics and get client gradient
    client_gradient = self.train_fn(self.model, self.client_loader,
                                    self.configs.optimizer, self.configs.criterion, self.configs.epochs, True)

    # for h in hook3_capture:
    #     h.remove()
    for h in hooks_capture_n:
        h.remove()

    post_step_running_statistics = []

    for module in self.model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.client = False
            post_step_running_statistics.append((module.running_mean.data.clone().detach(), module.running_var.data.clone().detach()))
    
    used_statistics = []

    for (rm_pre, rv_pre), (rm_post, rv_post) in zip(pre_step_running_statistics, post_step_running_statistics):
        used_mean = 10 * rm_post - 9 * rm_pre
        used_var = 10 * rv_post - 9 * rv_pre
        used_statistics.append((used_mean, used_var))

    # Put the used statistics as running statistics and set model to eval
    stat_idx = 0
    for module in self.model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            used_mean, used_var = used_statistics[stat_idx]
            module.running_mean.data.copy_(used_mean)
            n = bn_channel_element_counts[stat_idx]
            correction_factor = (n - 1) / n
            module.running_var.data.copy_(used_var * correction_factor)
            stat_idx += 1
    self.model.eval()

    (
        self.client_loader,
        self.original,
        self.reconstruction,
        self.reconstruction_labels,
        self.reconstruction_loader
    ) = self.configs.data_extension.get_at_data(self.client_loader)

    self.reconstruction.requires_grad = True
    self.client_gradient = [p.detach() for p in client_gradient]
    test_gradients = self.train_fn(self.model, self.client_loader,
                                    self.configs.optimizer, self.configs.criterion, self.configs.epochs, True)
    test_gradients = [p.detach() for p in test_gradients]
    for a,b in zip(self.client_gradient, test_gradients):
        print(f"allclose: {torch.allclose(a, b, atol=1e-4)}")


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

        # Add the TV loss term to penalize large variations between pixels, encouraging smoother images.
        rec_loss += self.configs.tv_reg * total_variation(self.reconstruction)
        # rec_loss += self.configs.l2_scale * l2_norm(self.reconstruction)
        rec_loss.backward()
        self.reconstruction.grad.sign_()
        return rec_loss
    return closure
