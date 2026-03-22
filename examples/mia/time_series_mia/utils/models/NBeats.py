# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
N-BEATS Model.
"""
from typing import Tuple

import numpy as np
import torch
from torch import nn


class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, input_dim, lookback, horizon, hidden_dim=128, num_layers=4, num_blocks=3):
        """
        Args:
        - input_dim: Number of features (variables).
        - lookback: Number of past time steps (input sequence length).
        - horizon: Number of future time steps to predict.
        - hidden_dim: Number of hidden units per layer.
        - num_layers: Number of layers per block.
        - num_blocks: Number of stacked blocks.
        """
        super(NBeats, self).__init__()
        self.init_params = {"input_dim": input_dim,
                    "lookback": lookback,
                    "horizon": horizon,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_blocks": num_blocks}
        if input_dim > 1:
            raise NotImplementedError("NBeats only works for univariate forecasting")
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        
        self.blocks = nn.ModuleList([NBeatsBlock(input_size=lookback,
                                                 theta_size=lookback + horizon,
                                                 basis_function=GenericBasis(backcast_size=lookback,
                                                                             forecast_size=horizon),
                                                 layers=num_layers,
                                                 layer_size=hidden_dim)
                                                 for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)
        residuals = x.flip(dims=(1,))
        forecast = torch.zeros(x.size(0), self.horizon, device=x.device, dtype=x.dtype)

        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        forecast = forecast.unsqueeze(-1)
        return forecast


class GenericBasis(nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float64) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=torch.float32),
            requires_grad=False)
        self.forecast_time = nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float64) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=torch.float32), requires_grad=False)

    def forward(self, theta: torch.Tensor):
        backcast = torch.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = torch.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float64),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float64) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float64)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float64)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = nn.Parameter(torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = nn.Parameter(torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = nn.Parameter(torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = nn.Parameter(torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32),
                                                    requires_grad=False)

    def forward(self, theta: torch.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = torch.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = torch.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = torch.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = torch.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast
