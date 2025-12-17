import math

import numpy as np
import torch
from torch import nn


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta=1.0):
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    return sigmas, alphas, alphas_prev

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps):
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    steps_out = ddim_timesteps + 1
    return steps_out

class InferenceModel(nn.Module):
    def __init__(self, x=None, batch_size=16):
        super(InferenceModel, self).__init__()
        if x is None:
            self.img = nn.Parameter(torch.randn(batch_size, 3, 64, 64))
        else:
            self.img = nn.Parameter(x)
        self.img.requires_grad = True
    def encode(self):
        return self.img

class GaussianDiffusion():
    """Gaussian Diffusion process with linear beta scheduling"""

    def __init__(self, T, schedule, ddim_timesteps=100, ddim_eta=1.0, b0=1e-4, bT=2e-2):
        # Diffusion steps
        self.T = T
        self.ddim_timesteps = ddim_timesteps

        # Noise schedule
        if schedule == "linear":
            b0 = b0
            bT = bT
            self.beta = np.linspace(b0, bT, T).astype(np.float32)

        elif schedule == "cosine":
            self.alphabar = self.cos_noise(np.arange(0, T+1, 1)) / self.cos_noise(0) # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

        self.betabar = np.cumprod(self.beta)
        self.alpha = 1 - self.beta
        self.alphabar = np.array(np.cumprod(self.alpha))

        # DDIM Parameters
        ddim_timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=ddim_timesteps,
                                                  num_ddpm_timesteps=self.T)

        self.ddim_sigmas, self.ddim_alphas, self.ddim_alphas_prev = make_ddim_sampling_parameters(
                                                                                    alphacums=torch.tensor(np.cumprod(self.alpha)).cpu(),
                                                                                    ddim_timesteps=ddim_timesteps,
                                                                                    eta=ddim_eta)

    def cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t/self.T + offset) / (1+offset)) ** 2

    def sample(self, x0, t):
        # Select noise scales
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
        atbar = torch.from_numpy(self.alphabar[t-1]).repeat(x0.shape[0],).view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), "Shape mismatch"

        # Sample noise and add to x0
        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1-atbar) * epsilon  # equation in line 4

        return xt, epsilon

    def inverse(self, net, x, start_t, w=1.0, y=None, device="cpu"):
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

    def inverse_ddim(self, net, x, start_t, w=1.0, y=None, device="cuda"):

        c = int(self.T / self.ddim_timesteps)
        ddim_start = int(start_t/c)
        time_range = range(ddim_start, 0, -1)

        for t in time_range:

            atbar = self.ddim_alphas[t-1]
            atbar_prev = self.ddim_alphas_prev[t-1]
            sigma_t = self.ddim_sigmas[t-1]

            t = torch.tensor([t*c]).view(1)
            e_t = w * net(x, t.float().to(device), y)[:,:3,:,:] - \
                        (w - 1) * net(x, t.float().to(device), torch.ones_like(y)*1000)[:,:3,:,:]

            pred_x0 = (x - np.sqrt(1-atbar) * e_t) / np.sqrt(atbar)
            dir_xt = np.sqrt(1. - atbar_prev - sigma_t**2) * e_t
            noise = sigma_t * torch.randn_like(x)
            x = np.sqrt(atbar_prev) * pred_x0 + dir_xt + noise

        return x
