"""Run script."""
from dataclasses import dataclass
from typing import Callable

from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader

from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients


@dataclass
class InvertingConfig():
    total_varation = 1e-6

def run_inverting(model: Module, client_data: DataLoader, train_fn: Callable,
                data_mean:Tensor, data_std: Tensor, config: dict) -> None:
    """Runs InvertingGradients."""
    attack = InvertingGradients(model, client_data, train_fn, data_mean, data_std, config)
    result = attack.run_attack()
    result.prepare_privacy_risk_report("InvertingGradients", "./leakpro_output/results")

def run_inverting_noggrant(model, tensordataset, train_fn):
    pass
    # NOT SAFE
    # setting 3 we start having good priavxy
    # GO TOWARDS MORE SAFE SettiNGS
    # run pre-training of model (train on all data)
    # run with 1 client img
    # run with 10 client img
    # run with 4 epochs
    # run with tv = 1e4
    # run with tv = 1e6

def run_gia_noggrant():
    pass
