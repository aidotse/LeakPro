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
from leakpro import shadow_models
from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.dataset import get_dataloader
from leakpro.dev_utils.data_preparation import get_adult_dataset, get_cifar10_dataset, prepare_train_test_datasets
from leakpro.reporting.utils import prepare_priavcy_risk_report
from leakpro.utils.input_handler import get_class_from_module, import_module_from_file


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

    retrain = False
    # Create the population dataset and target_model
    if "adult" in configs["data"]["dataset"]:
        population = get_adult_dataset(configs["data"]["dataset"], configs["data"]["data_dir"], logger)
        target_model = shadow_models.NN(configs["train"]["inputs"], configs["train"]["outputs"])
    elif "cifar10" in configs["data"]["dataset"]:
        population = get_cifar10_dataset(configs["data"]["dataset"], configs["data"]["data_dir"], logger)
        target_model = shadow_models.ConvNet()

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

    # Get the target  metadata
    target_model_metadata_path = f"{configs["target"]["trained_model_metadata_path"]}"
    with open(target_model_metadata_path, "rb") as f:
        target_model_metadata = joblib.load(f)

    # Create a class instance of target model
    target_module = import_module_from_file(configs["target"]["module_path"])
    target_model_blueprint = get_class_from_module(target_module, configs["target"]["model_class"])
    logger.info(f"Target model blueprint created from {configs['target']['model_class']} in {configs['target']['module_path']}")

    # Load the target model parameters into the blueprint
    with open(configs["target"]["trained_model_path"], "rb") as f:
        target_model = target_model_blueprint(**target_model_metadata["model_metadata"]["init_params"])
        target_model.load_state_dict(torch.load(f))
        logger.info(f"Loaded target model from {configs['target']['trained_model_path']}")

    # Get the population dataset
    with open(configs["target"]["data_path"], "rb") as file:
        population = joblib.load(file)
        logger.info(f"Loaded population dataset from {configs['target']['data_path']}")
    # ------------------------------------------------
    # Now we have the target model, its metadata, and the train/test dataset indices.
    attack_scheduler = AttackScheduler(
        population,
        target_model,
        target_model_metadata["model_metadata"],
        configs,
        logger,
    )
    audit_results = attack_scheduler.run_attacks()

    report_log = configs["audit"]["report_log"]
    privacy_game = configs["audit"]["privacy_game"]
    n_shadow_models = configs["audit"]["num_shadow_models"]
    n_attack_data_size = configs["audit"]["f_attack_data_size"]

    prepare_priavcy_risk_report(
            report_dir,
            [audit_results["qmia"]["result_object"]],
            configs["audit"],
            save_path=f"{report_dir}/{report_log}/{privacy_game}/ns_{n_shadow_models}_fs_{n_attack_data_size}",
        )
