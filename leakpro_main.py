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

import dev_utils.train as utils
from dev_utils import shadow_model_blueprints
from dev_utils.cifar10_input_handler import Cifar10InputHandler
from dev_utils.data_preparation import (
    get_adult_dataset,
    get_cifar10_dataset,
    get_cifar100_dataset,
    get_cinic10_dataset,
    prepare_train_test_datasets,
)
from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.dataset import get_dataloader
from leakpro.reporting.utils import prepare_privacy_risk_report
from leakpro.utils.handler_logger import setup_log


def generate_user_input(configs: dict, retrain: bool = False, logger: logging.Logger = None)->None:
    """Generate user input for the target model."""
    # ------------------------------------------------

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
    generate_user_input(user_configs, retrain=True, logger=logger) # This is for developing purposes only

    start_time = time.time()
    # ------------------------------------------------
    # LEAKPRO starts here
    args = "./config/audit.yaml"
    with open(args, "rb") as f:
        configs = yaml.safe_load(f)

    # Set the random seed, log_dir and inference_game
    manual_seed(configs["audit"]["random_seed"])
    np.random.seed(configs["audit"]["random_seed"])
    random.seed(configs["audit"]["random_seed"])

    # Create directory to store results
    report_dir = f"{configs['audit']['report_log']}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # Create user input handler
    handler = Cifar10InputHandler(configs=configs, logger=logger)

    attack_scheduler = AttackScheduler(handler)
    audit_results = attack_scheduler.run_attacks()

    for attack_name in audit_results:
        logger.info(f"Preparing results for attack: {attack_name}")

        prepare_privacy_risk_report(
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
