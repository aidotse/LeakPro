"""Main script to run LEAKPRO on a target model."""

import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

import leakpro.dev_utils.train as utils
from leakpro import shadow_model_blueprints
from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.dataset import get_dataloader
from leakpro.dev_utils.data_preparation import (
    get_adult_dataset,
    get_cifar10_dataset,
    get_cifar100_dataset,
    get_cinic10_dataset,
    prepare_train_test_datasets,
)
from leakpro.reporting.utils import prepare_priavcy_risk_report
from leakpro.user_inputs.cifar10_input_handler import Cifar10InputHandler


def setup_log(name: str, save_file: bool=True) -> logging.Logger:
    """Generate the logger for the current run.

    Args:
    ----
        name (str): Logging file name.
        save_file (bool): Flag about whether to save to file.

    Returns:
    -------
        logging.Logger: Logger object for the current run.

    """
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    # Console handler for output to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    my_logger.addHandler(console_handler)

    if save_file:
        filename = f"log_{name}.log"
        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)
        my_logger.addHandler(log_handler)

    return my_logger

def generate_user_input(configs: dict, logger: logging.Logger)->None:
    """Generate user input for the target model."""
    # ------------------------------------------------

    retrain = True
    # Create the population dataset and target_model
    if configs["data"]["dataset"] == "adult":
        population = get_adult_dataset(configs["data"]["dataset"], configs["data"]["data_dir"], logger)
        target_model = shadow_model_blueprints.NN(configs["train"]["inputs"], configs["train"]["outputs"])
    elif configs["data"]["dataset"] == "cifar10":
        population = get_cifar10_dataset(configs["data"]["dataset"], configs["data"]["data_dir"], logger)
        target_model = shadow_model_blueprints.ResNet18(configs["train"]["num_classes"])
    elif configs["data"]["dataset"] == "cifar100":
        population = get_cifar100_dataset(configs["data"]["dataset"], configs["data"]["data_dir"], logger)
        target_model = shadow_model_blueprints.ResNet18(configs["train"]["num_classes"])
    elif configs["data"]["dataset"] == "cinic10":
        population = get_cinic10_dataset(configs["data"]["dataset"], configs["data"]["data_dir"], logger)
        target_model = shadow_model_blueprints.ResNet18(configs["train"]["num_classes"])

    n_population = len(population)

    # Create target training dataset and test dataset
    # NOTE: this should not be done as the model is provided by the user
    train_test_dataset = prepare_train_test_datasets(n_population, configs["data"])

    train_loader = get_dataloader(
        Subset(population, train_test_dataset["train_indices"]),
        batch_size=configs["train"]["batch_size"],
        shuffle=True,
    )
    test_loader = get_dataloader(
        Subset(population, train_test_dataset["test_indices"]),
        batch_size=configs["train"]["test_batch_size"],
    )

    if retrain:
        target_model = utils.train(target_model, train_loader, configs, test_loader, train_test_dataset, logger)


if __name__ == "__main__":

    user_args = "./config/dev_config/cifar10.yaml" # noqa: ERA001

    with open(user_args, "rb") as f:
        user_configs = yaml.safe_load(f)

    # Setup logger
    logger = setup_log("LeakPro", save_file=True)

    # Generate user input
    generate_user_input(user_configs, logger) # This is for developing purposes only

    start_time = time.time()
    # ------------------------------------------------
    # LEAKPRO starts here
    args = "./config/audit.yaml"
    with open(args, "rb") as f:
        configs = yaml.safe_load(f)

    # # Set the random seed, log_dir and inference_game
    # manual_seed(configs["audit"]["random_seed"])
    # np.random.seed(configs["audit"]["random_seed"])
    # random.seed(configs["audit"]["random_seed"])
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            # Ensure deterministic behavior on GPUs
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Create directory to store results
    report_dir = f"{configs['audit']['report_log']}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # Create user input handler
    handler = Cifar10InputHandler(configs=configs, logger=logger)

    attack_scheduler = AttackScheduler(handler)
    audit_results = attack_scheduler.run_attacks()

    for attack_name in audit_results:
        logger.info(f"Preparing results for attack: {attack_name}")

        prepare_priavcy_risk_report(
                audit_results[attack_name]["result_object"],
                configs["audit"],
                save_path=f"{report_dir}/{attack_name}",
            )
    # ------------------------------------------------
    # Save the configs and user_configs
    config_log_path = configs["audit"]["config_log"]
    os.makedirs(config_log_path, exist_ok=True)
    with open(f"{config_log_path}/audit.yaml", "w") as f:
        yaml.safe_dump(configs, f)
    with open(f"{config_log_path}/user_config.yaml", "w") as f:
        yaml.safe_dump(user_configs, f)
