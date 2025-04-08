"""Utility functions for converting PyTorch objects to LeakPro schemas."""

import inspect

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler

from leakpro.schemas import DataLoaderConfig, LossConfig, OptimizerConfig


def _loss_to_config(loss_fn: Module) -> LossConfig:
    """Convert a PyTorch loss function to a LossConfig instance."""

    loss_name = loss_fn.__class__.__name__.lower()  # Convert class name to lowercase
    params = {k: getattr(loss_fn, k) for k in vars(loss_fn) if not k.startswith("_")}
    return LossConfig(name=loss_name, params=params)

def _optimizer_to_config(optimizer: Optimizer) -> OptimizerConfig:
    """Convert a PyTorch optimizer to an OptimizerConfig instance."""

    optimizer_name = optimizer.__class__.__name__.lower()
    params = {k: v for k, v in optimizer.defaults.items() if not k.startswith("_")}

    return OptimizerConfig(name=optimizer_name, params=params)

def _dataloader_to_config(dataloader: DataLoader) -> DataLoaderConfig:
    """Convert a PyTorch DataLoader to a DataLoaderConfig instance, excluding dataset and batch_sampler."""

    # Exclude 'dataset' and 'batch_sampler' from stored parameters
    exclude_keys = {"dataset", "sampler", "batch_sampler"}
    init_signature = inspect.signature(DataLoader.__init__)  # Get constructor signature
    param_names = [name for name in list(init_signature.parameters.keys())[1:] if name not in exclude_keys]
    params = {name: getattr(dataloader, name) for name in param_names if hasattr(dataloader, name)}
    params["shuffle"] = isinstance(dataloader.sampler, RandomSampler) # this is stored in sampler so must be handled separately

    return DataLoaderConfig(params=params)



def _get_model_init_params(model: Module) -> dict:
    """Extracts the parameters that were passed to the __init__ method from an object instance."""

    cls = model.__class__
    init_signature = inspect.signature(cls.__init__)  # Get constructor signature
    param_names = list(init_signature.parameters.keys())[1:]  # Skip 'self'

    # Extract only parameters that exist in instance attributes
    return {name: getattr(model, name) for name in param_names if hasattr(model, name)}


