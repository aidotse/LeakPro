"""Main script to run LEAKPRO on a target model."""

import logging
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

import leakpro.dev_utils.train as utils
from leakpro import shadow_model_blueprints
from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.dataset import get_dataloader
from leakpro.dev_utils.data_preparation import (
    get_adult_dataset,
    get_cifar10_dataset,
    get_cinic10_dataset,
    prepare_train_test_datasets,
)
from leakpro.reporting.utils import prepare_priavcy_risk_report
from leakpro.utils.input_handler import get_class_from_module, import_module_from_file
from leakpro.user_code.user_definitions import Cifar10CodeHandler

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
    if "adult" in configs["data"]["dataset"]:
        population = get_adult_dataset(configs["data"]["dataset"], configs["data"]["data_dir"], logger)
        target_model = shadow_model_blueprints.NN(configs["train"]["inputs"], configs["train"]["outputs"])
    elif "cifar10" in configs["data"]["dataset"]:
        population = get_cifar10_dataset(configs["data"]["dataset"], configs["data"]["data_dir"], logger)
        target_model = shadow_model_blueprints.ConvNet()
    elif "cinic10" in configs["data"]["dataset"]:
        population = get_cinic10_dataset(configs["data"]["dataset"], configs["data"]["data_dir"], logger)
        target_model = shadow_model_blueprints.ConvNet()

    n_population = len(population)

    # Create target training dataset and test dataset
    # NOTE: this should not be done as the model is provided by the user
    train_test_dataset = prepare_train_test_datasets(n_population, configs["data"])

    train_loader = get_dataloader(
        torch.utils.data.Subset(population, train_test_dataset["train_indices"]),
        batch_size=configs["train"]["batch_size"],
        shuffle=True,
    )
    test_loader = get_dataloader(
        torch.utils.data.Subset(population, train_test_dataset["test_indices"]),
        batch_size=configs["train"]["test_batch_size"],
    )

    if retrain:
        target_model = utils.train(target_model, train_loader, configs, test_loader, train_test_dataset, logger)


if __name__ == "__main__":


    #args = "./config/adult.yaml"  # noqa: ERA001
    user_args = "./config/dev_config/cifar10.yaml" # noqa: ERA001
    # user_args = "./config/dev_config/cinic10.yaml" # noqa: ERA001

    with open(user_args, "rb") as f:
        user_configs = yaml.safe_load(f)

    # Setup logger
    logger = setup_log("analysis")

    # Generate user input
    generate_user_input(user_configs, logger) # This is for developing purposes only

    start_time = time.time()
    # ------------------------------------------------
    # LEAKPRO starts here
    args = "./config/audit.yaml" # noqa: ERA001
    with open(args, "rb") as f:
        configs = yaml.safe_load(f)

    # Set the random seed, log_dir and inference_game
    torch.manual_seed(configs["audit"]["random_seed"])
    np.random.seed(configs["audit"]["random_seed"])
    random.seed(configs["audit"]["random_seed"])

    # Create directory to store results
    report_dir = f"{configs['audit']['report_log']}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # Create user input handler
    # TODO: make this not hardcoded. Maybe add to configs and have one option "custom" which loads an arbitrary user file
    handler = Cifar10CodeHandler(configs=configs, logger=logger)
    handler.setup()
    # ------------------------------------------------
    # Now we have the target model, its metadata, and the train/test dataset indices (all of this is defined in handler)
    attack_scheduler = AttackScheduler(
        handler,
        configs,
        logger,
    )
    audit_results = attack_scheduler.run_attacks()

    for attack_name in audit_results:
        logger.info(f"Preparing results for attack: {attack_name}")

        prepare_priavcy_risk_report(
                audit_results[attack_name]["result_object"],
                configs["audit"],
                save_path=f"{report_dir}/{attack_name}",
            )
