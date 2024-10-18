"""Run script."""
from typing import Callable

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients


def run_inverting(model: Module, client_data: DataLoader, train_fn: Callable,
                data_mean:Tensor, data_std: Tensor, config: dict) -> None:
    """Runs InvertingGradients."""
    attack = InvertingGradients(model, client_data, train_fn, data_mean, data_std, config)
    result = attack.run_attack()
    result.prepare_privacy_risk_report("InvertingGradients", "./leakpro_output/results")

def run_inverting_audit(model: Module, tensordataset: TensorDataset, train_fn: Callable) -> None:
    """Runs an audit for InvertingGradients with different parameters and pre-training."""
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

