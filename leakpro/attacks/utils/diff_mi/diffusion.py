"""Diffusion utilities for Diff-MI."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


def make_ddim_sampling_parameters(
    alphacums: np.ndarray,
    ddim_timesteps: np.ndarray,
    eta: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute DDIM sampling parameters."""
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    return sigmas, alphas, alphas_prev

def make_ddim_timesteps(
    ddim_discr_method: str,
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
) -> np.ndarray:
    """Create the discretized DDIM timesteps."""
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        c = max(c, 1)
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise ValueError(f"Unsupported ddim_discr_method: {ddim_discr_method}")
    # Keep indices in [0, num_ddpm_timesteps - 1] for direct array indexing.
    return np.clip(ddim_timesteps, 0, num_ddpm_timesteps - 1)

class InferenceModel(nn.Module):
    """Inference model holding the image tensor for optimization."""

    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        batch_size: int = 16,
        data_channels: int = 3,
        data_height: int = 64,
        data_width: int = 64,
    ) -> None:
        """Initialize the optimization image tensor.

        Args:
        ----
            x: Optional initial image tensor.
            batch_size: Number of images to optimize when `x` is not provided.
            data_channels: Number of image channels.
            data_height: Image height.
            data_width: Image width.

        """
        super().__init__()
        if x is None:
            self.img = nn.Parameter(torch.randn(batch_size, data_channels, data_height, data_width))
        else:
            self.img = nn.Parameter(x)
        self.img.requires_grad = True

    def encode(self) -> torch.Tensor:
        """Return the current image tensor."""
        return self.img

class GaussianDiffusion:
    """Gaussian Diffusion process with linear beta scheduling."""

    def __init__(
        self,
        num_steps: int,
        schedule: str,
        ddim_timesteps: int = 100,
        ddim_eta: float = 1.0,
        b0: float = 1e-4,
        b_t: float = 2e-2,
    ) -> None:
        """Initialize the diffusion schedule and DDIM helper values.

        Args:
        ----
            num_steps: Number of diffusion steps.
            schedule: Noise schedule name.
            ddim_timesteps: Number of DDIM steps used for inversion.
            ddim_eta: DDIM noise scale.
            b0: Initial beta value for the linear schedule.
            b_t: Final beta value for the linear schedule.

        """
        # Diffusion steps
        self.T = num_steps
        self.ddim_timesteps = max(1, min(ddim_timesteps, num_steps))

        # Noise schedule
        if schedule == "linear":
            self.beta = np.linspace(b0, b_t, num_steps).astype(np.float32)

        elif schedule == "cosine":
            self.alphabar = self.cos_noise(np.arange(0, num_steps + 1, 1)) / self.cos_noise(0)  # extra alpha for b_t
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

        self.betabar = np.cumprod(self.beta)
        self.alpha = 1 - self.beta
        self.alphabar = np.array(np.cumprod(self.alpha))

        # DDIM Parameters
        ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method="uniform",
            num_ddim_timesteps=self.ddim_timesteps,
            num_ddpm_timesteps=self.T,
        )

        self.ddim_sigmas, self.ddim_alphas, self.ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=torch.tensor(np.cumprod(self.alpha)).cpu(),
            ddim_timesteps=ddim_timesteps,
            eta=ddim_eta,
        )

    def cos_noise(self, t: np.ndarray) -> np.ndarray:
        """Cosine noise schedule helper."""
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t/self.T + offset) / (1+offset)) ** 2

    def sample(self, x0: torch.Tensor, t: Union[np.ndarray, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a noisy x_t from x0 at timestep t."""
        # Select noise scales
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
        atbar = torch.from_numpy(self.alphabar[t-1]).repeat(x0.shape[0],).view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), "Shape mismatch"

        # Sample noise and add to x0
        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1-atbar) * epsilon  # equation in line 4

        return xt, epsilon

    def inverse(
        self,
        net: torch.nn.Module,
        x: torch.Tensor,
        start_t: int,
        w: float = 1.0,
        y: Optional[torch.Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> torch.Tensor:
        """Run the reverse diffusion process."""
        _ = w
        # Specify starting conditions and number of steps to run for

        for t in range(start_t, 0, -1):
            at = self.alpha[t-1]
            atbar = self.alphabar[t-1]

            if t > 1:
                z = torch.randn_like(x)
                atbar_prev = self.alphabar[t-2]
                beta_tilde = self.beta[t-1] * (1 - atbar_prev) / (1 - atbar)
            else:
                z = torch.zeros_like(x)
                beta_tilde = 0

            with torch.no_grad():
                t = torch.tensor([t]).view(1)
                pred = net(x, t.float().to(device), y)[:,:3,:,:]

            x = (1 / np.sqrt(at)) * (x - ((1-at) / np.sqrt(1-atbar)) * pred) + np.sqrt(beta_tilde) * z

        return x

    def inverse_ddim(
        self,
        net: torch.nn.Module,
        x: torch.Tensor,
        start_t: int,
        w: float = 1.0,
        y: Optional[torch.Tensor] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """Run the DDIM reverse process."""
        c = max(int(self.T / self.ddim_timesteps), 1)
        ddim_start = max(int(math.ceil(start_t / c)), 1)
        time_range = range(ddim_start, 0, -1)

        if y is not None:
            uncond_idx = getattr(net, "num_classes", None)
            uncond_y = torch.full_like(y, fill_value=int(uncond_idx) - 1) if uncond_idx is not None else y

        for t in time_range:

            atbar = self.ddim_alphas[t-1]
            atbar_prev = self.ddim_alphas_prev[t-1]
            sigma_t = self.ddim_sigmas[t-1]

            t = torch.tensor([t*c]).view(1)
            if y is None:
                e_t = net(x, t.float().to(device), y)[:,:3,:,:]
            else:
                e_t = w * net(x, t.float().to(device), y)[:,:3,:,:] - \
                            (w - 1) * net(x, t.float().to(device), uncond_y)[:,:3,:,:]

            pred_x0 = (x - np.sqrt(1-atbar) * e_t) / np.sqrt(atbar)
            dir_xt = np.sqrt(1. - atbar_prev - sigma_t**2) * e_t
            noise = sigma_t * torch.randn_like(x)
            x = np.sqrt(atbar_prev) * pred_x0 + dir_xt + noise

        return x
