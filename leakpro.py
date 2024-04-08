"""Main script to run LEAKPRO on a target model."""

import logging
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

import leakpro.train as util
from leakpro import dataset, models
from leakpro.mia_attacks.attack_scheduler import AttackScheduler
from leakpro.reporting.utils import prepare_priavcy_risk_report


def setup_log(name: str, save_file: bool) -> logging.Logger:
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


if __name__ == "__main__":

    RETRAIN = False
    # args = "./config/adult.yaml"  # noqa: ERA001
    args = "./config/cifar10.yaml" # noqa: ERA001
    with open(args, "rb") as f:
        configs = yaml.safe_load(f)

    # Set the random seed, log_dir and inference_game
    torch.manual_seed(configs["run"]["random_seed"])
    np.random.seed(configs["run"]["random_seed"])
    random.seed(configs["run"]["random_seed"])

    # Setup logger
    log_dir = configs["run"]["log_dir"]
    logger = setup_log("time_analysis", configs["run"]["time_log"])

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    report_dir = f"{log_dir}/{configs['audit']['report_log']}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ------------------------------------------------
    # Create the population dataset
    population = dataset.get_dataset(configs["data"]["dataset"], configs["data"]["data_dir"], logger)
    N_population = len(population)

    # Create target training dataset and test dataset
    # NOTE: this should not be done as the model is provided by the user
    train_test_dataset = dataset.prepare_train_test_datasets(N_population, configs["data"])

    train_loader = dataset.get_dataloader(
        torch.utils.data.Subset(population, train_test_dataset["train_indices"]),
        batch_size=configs["train"]["batch_size"],
        shuffle=True,
    )
    test_loader = dataset.get_dataloader(
        torch.utils.data.Subset(population, train_test_dataset["test_indices"]),
        batch_size=configs["train"]["test_batch_size"],
    )

    # Train the target model
    if "adult" in configs["data"]["dataset"]:
        model = models.NN(configs["train"]["inputs"], configs["train"]["outputs"])
    elif "cifar10" in configs["data"]["dataset"]:
        model = models.ConvNet()
    if RETRAIN:
        model = util.train(model, train_loader, configs, test_loader, train_test_dataset, logger)


    # ------------------------------------------------
    # LEAKPRO starts here
    # Read in model, population, and metadata
    data_dir = configs["data"]["data_dir"]
    data_file = configs["data"]["dataset"]
    dataset_path = f"{data_dir}/{data_file}.pkl"
    with open(dataset_path, "rb") as file:
        population = joblib.load(file)

    # Get the training and test data
    train_test_data = train_test_dataset

    # Get the target model + metadata
    target_model_metadata_path = f"{log_dir}/models_metadata.pkl"
    with open(target_model_metadata_path, "rb") as f:
        target_model_metadata = joblib.load(f)
    target_model_path = f"{log_dir}/model_0.pkl"
    with open(target_model_path, "rb") as f:
        if "adult" in configs["data"]["dataset"]:
            target_model = models.NN(
                configs["train"]["inputs"], configs["train"]["outputs"]
            )  # TODO: read metadata to get the model
        elif "cifar10" in configs["data"]["dataset"]:
            target_model = models.ConvNet()
        target_model.load_state_dict(torch.load(f))

    # ------------------------------------------------
    # Now we have the target model, its metadata, and the train/test dataset
    # indices.
    attack_scheduler = AttackScheduler(
        population,
        train_test_dataset,
        target_model,
        target_model_metadata,
        configs,
        log_dir,
        logger,
    )  # TODO metadata includes indices for train and test data
    audit_results = attack_scheduler.run_attacks()

    attack_name = str(configs["audit"]["attack_list"][0])
    logger.info(str(audit_results[attack_name]["result_object"]))

    prepare_priavcy_risk_report(
            log_dir,
            [audit_results[attack_name]["result_object"]],
            configs["audit"],
            save_path=f"{log_dir}/{configs['audit']['report_log']}/{configs['audit']['privacy_game']}/ns_{configs['audit']['num_shadow_models']}_fs_{configs['audit']['f_attack_data_size']}",
        )
