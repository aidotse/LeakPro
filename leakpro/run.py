"""Run script."""
import os
from collections import OrderedDict
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, random_split

from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.utils.logger import logger


def run_inverting(model: Module, client_data: DataLoader, train_fn: Callable,
                data_mean:Tensor, data_std: Tensor, config: dict, experiment_name: str = "InvertingGradients") -> None:
    """Runs InvertingGradients."""
    attack = InvertingGradients(model, client_data, train_fn, data_mean, data_std, config)
    result = attack.run_attack()
    result.prepare_privacy_risk_report(experiment_name, "./leakpro_output/results")

def run_inverting_audit(model: Module, dataset: Dataset,
                        train_fn: Callable, data_mean: torch.Tensor, data_std: torch.Tensor
                        ) -> None:
    """Runs a thourough audit for InvertingGradients with different parameters and pre-training.

    Parameters
    ----------
    model: Module
        Starting model that has not been exposed to the tensordataset.
    dataset: Dataset
        Your full dataset containg all unseen data points.
    train_fn: Callable
        A Meta training function which uses an metaoptimizer to simulate training steps without moving the model.
    data_mean: Optional[torch.Tensor]
        Mean of the dataset. Will try to infer it if not supplied.
    data_std: Optional[torch.Tensor]
        STD of the dataset. Will try to infer it if not supplied.

    """
    # Randomly split the dataset: 100 random images for attack, rest for pre-training
    total_images = len(dataset)
    config = InvertingConfig()

    # Prepare for the inverting attack experiments
    experiment_configs = [
        (1, 1),   # 1 batch of 1 image
        (1, 4),   # 1 batch of 4 images
        (1, 16),  # 1 batch of 16 images
        (1, 32),  # 1 batch of 32 images
        (2, 16),  # 2 batches of 16 images
        (4, 8)    # 4 batches of 8 images
    ]

    total_variations = [1.0e-03, 1.0e-05, 1.0e-07]

    epochs_config = [1 , 4]

    # Perform attack with varying (num_batches, batch_size, total_variation, epochs)
    for num_batches, batch_size in experiment_configs:
        # Create a dataloader for attack using the specified number of batches and batch size
        client_data, _ = random_split(dataset, [num_batches * batch_size, total_images - num_batches * batch_size])
        client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=False)
        for tv in total_variations:
            config.total_variation = tv
            for epochs in epochs_config:
                config.epochs = epochs
                # Run the inverting attack with current client_loader and config
                experiment_name = "Inverting_batch_size_"+str(num_batches)+"num_batches_"+str(num_batches) \
                                +"_epochs_" + str(epochs) + "_tv_" + str(tv)
                logger.info(f"Running experiment: {experiment_name}")
                run_inverting(model, client_loader, train_fn, data_mean, data_std, config,
                            experiment_name=experiment_name)

