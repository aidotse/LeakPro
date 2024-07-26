"""Main script to run LEAKPRO on a target model."""

import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml
from torch import manual_seed
from torch.utils.data import Subset

from leakpro import shadow_model_blueprints
from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.dataset import get_dataloader
from leakpro.dev_utils.data_preparation import (
    get_adult_dataset,
    get_cifar10_dataset,
    get_cinic10_dataset,
)
from leakpro.fl_utils.client_data import prepare_fl_client_dataset
from leakpro.reporting.utils import prepare_priavcy_risk_report
from leakpro.user_inputs.cifar10_gia_input_handler import Cifar10GIAInputHandler


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

def generate_client_input(configs: dict, logger: logging.Logger) -> tuple:
    """Generate client data and data splits."""
    # ------------------------------------------------

    # Create the population dataset and target_model
    if "adult" in configs["gia_settings"]["dataset"]:
        population = get_adult_dataset(configs["gia_settings"]["dataset"], "./target/data", logger)
        target_model = shadow_model_blueprints.NN("not implemented")
    elif "cifar10" in configs["gia_settings"]["dataset"]:
        population = get_cifar10_dataset(configs["gia_settings"]["dataset"], "./target/data", logger)
        target_model = shadow_model_blueprints.ConvNet()
    elif "cinic10" in configs["gia_settings"]["dataset"]:
        population = get_cinic10_dataset(configs["gia_settings"]["dataset"], "./target/data", logger)
        target_model = shadow_model_blueprints.ResNet18(num_classes=10)

    n_population = len(population)

    train_dataset, _ = prepare_fl_client_dataset(n_population, configs)

    client_loader = get_dataloader(
        Subset(population, train_dataset),
        batch_size=configs["gia_settings"]["client_batch_size"],
        shuffle=True,
    )

    return client_loader, target_model

if __name__ == "__main__":

    # Setup logger
    logger = setup_log("LeakPro", save_file=True)

    start_time = time.time()
    # ------------------------------------------------
    # LEAKPRO starts here
    args = "./config/audit.yaml" # noqa: ERA001

    with open(args, "rb") as f:
        configs = yaml.safe_load(f)

    # Create client loader and model
    client_loader, target_model = generate_client_input(configs["audit"], logger)
    # Set the random seed, log_dir
    manual_seed(configs["audit"]["random_seed"])
    np.random.seed(configs["audit"]["random_seed"])
    random.seed(configs["audit"]["random_seed"])

    # Create directory to store results
    report_dir = f"{configs['audit']['report_log']}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # Create user input handler
    handler = Cifar10GIAInputHandler(configs=configs, logger=logger, client_data=client_loader,target_model=target_model)

    attack_scheduler = AttackScheduler(handler)
    audit_results = attack_scheduler.run_attacks()

    for attack_name in audit_results:
        logger.info(f"Preparing results for attack: {attack_name}")

        audit_results[attack_name]["result_object"].prepare_privacy_risk_report(attack_name, report_dir)
    # ------------------------------------------------
    # Save the configs and user_configs
    config_log_path = configs["audit"]["config_log"]
    os.makedirs(config_log_path, exist_ok=True)
    with open(f"{config_log_path}/audit.yaml", "w") as f:
        yaml.safe_dump(configs, f)
