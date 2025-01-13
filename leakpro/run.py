"""Run script."""
import copy
from typing import Callable

from leakpro.utils.seed import seed_everything
import optuna
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, random_split

from leakpro.attacks.gia_attacks.huang import Huang, HuangConfig
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.utils.logger import logger


def run_huang(model: Module, client_data: DataLoader, train_fn: Callable,
                data_mean:Tensor, data_std: Tensor, config: dict, experiment_name: str = "Huang",
                path:str = "./leakpro_output/results", save:bool = True) -> None:
    """Runs Huang."""
    attack = Huang(model, client_data, train_fn, data_mean, data_std, config)
    gen = attack.run_attack()
    try:
        while True:
            _, _ = next(gen)
    except StopIteration as e:
        result = e.value
    if save:
        result.save(name=experiment_name, path=path, config=config)
    return result

def huang_optuna(model: Module, client_dataloader: DataLoader, train_fn: Callable,
                 data_mean: Tensor, data_std:Tensor, seed:int =1234) -> None:
    """Runs Evaluating with Huang et al., using optuna for finding optimal hyperparameters."""
    def objective(trial: optuna) -> Tensor:
        total_variation = trial.suggest_loguniform("total_variation", 1e-6, 1e-1)
        bn_reg = trial.suggest_loguniform("bn_reg", 1e-4, 1e-1)

        # Reproducibility
        seed_everything(seed)

        # Deepcopy the model for this trial
        trial_model = copy.deepcopy(model)

        # Configurations for the attack
        configs = HuangConfig()
        configs.total_variation = total_variation
        configs.bn_reg = bn_reg

        # Create the attack instance
        attack = Huang(trial_model, client_dataloader, train_fn, data_mean, data_std, configs)

        # Run the attack
        gen = attack.run_attack()
        try:
            while True:
                step, psnr = next(gen)
                trial.report(psnr, step)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        except StopIteration as e:
            gia_result = e.value

        # Save final results if the trial isn't pruned
        gia_result.save(name="Huang_Optuna", path="./leakpro_output/results", config=configs)
        return gia_result.SSIM_score

    # Define the pruner and study
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    # Run optimization
    study.optimize(objective, n_trials=50)

    # Display and save the results
    logger.info("Best hyperparameters:", study.best_params)
    logger.info("Best PSNR:", study.best_value)

    results_file = "optuna_results.txt"
    with open(results_file, "w") as f:
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Results saved to {results_file}")

def run_inverting(model: Module, client_data: DataLoader, train_fn: Callable,
                data_mean:Tensor, data_std: Tensor, config: dict, experiment_name: str = "InvertingGradients",
                path:str = "./leakpro_output/results", save:bool = True) -> None:
    """Runs InvertingGradients."""
    attack = InvertingGradients(model, client_data, train_fn, data_mean, data_std, config)
    result = attack.run_attack()
    if save:
        result.save(name=experiment_name, path=path, config=config)
    return result

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

    total_variations = [1.0e-04, 1.0e-05, 1.0e-06]

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
                experiment_name = "Inverting_batch_size_"+str(batch_size)+"num_batches_"+str(num_batches) \
                                +"_epochs_" + str(epochs) + "_tv_" + str(tv)
                logger.info(f"Running experiment: {experiment_name}")
                run_inverting(model, client_loader, train_fn, data_mean, data_std, config,
                            experiment_name=experiment_name)

