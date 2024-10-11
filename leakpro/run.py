"""Run script."""
from typing import Callable

from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset

from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients


def run_geiping(model: Module, client_data: TensorDataset, train_fn: Callable,
                data_mean:Tensor, data_std: Tensor, config: dict) -> None:
    """Runs InvertingGradients."""
    attack = InvertingGradients(model, client_data, train_fn, data_mean, data_std, config)
    result = attack.run_attack()
    result.prepare_privacy_risk_report("InvertingGradients", "./leakpro_output/results")
