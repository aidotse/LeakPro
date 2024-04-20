"""Main script to run LEAKPRO on a target model."""

import logging
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from torch import nn

import leakpro.dev_utils.train as utils
from leakpro import shadow_models
from leakpro.dataset import get_dataloader
from leakpro.dev_utils.data_preparation import get_adult_dataset, get_cifar10_dataset, prepare_train_test_datasets
from leakpro.mia_attacks.attack_scheduler import AttackScheduler
from leakpro.reporting.utils import prepare_priavcy_risk_report


def create_model(
    current_instance: nn.Module = None,
    new_class: type = None,
    metadata: dict = None,
    *args,
    **kwargs
) -> nn.Module:
    """
    Create a new model instance, either by copying the class from an existing instance
    or by using a specified class. If metadata is provided, it can be used to extract
    default initialization parameters.

    Args:
        current_instance (nn.Module): An existing model instance to infer the class from.
        new_class (type): A specific class to use for creating the new model.
        metadata (dict): Metadata containing initialization parameters.
        *args: Additional positional arguments for model initialization.
        **kwargs: Additional keyword arguments for model initialization.

    Returns:
        nn.Module: A new instance of the specified or inferred model class.
    """

    # If new_class is not specified, infer it from the current_instance
    if new_class is None:
        if current_instance is not None:
            new_class = type(current_instance)
        else:
            raise ValueError("Either new_class or current_instance must be provided.")

    # Ensure the provided or inferred class is a subclass of nn.Module
    if not issubclass(new_class, nn.Module):
        raise ValueError("Provided class must be a subclass of torch.nn.Module")

    # Use metadata to get default initialization parameters, if provided
    if metadata is not None and new_class in metadata:
        params = metadata[new_class]['parameters']
        # Get default values for initialization
        init_args = {param: details['default'] for param, details in params.items()}
        # Merge init_args with additional arguments and keyword arguments
        return new_class(*args, **{**init_args, **kwargs})

    # If no metadata, just create the model with given args and kwargs
    return new_class(*args, **kwargs)

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
    log_dir = configs["run"]["log_dir"]
    target_model_metadata_path = f"{log_dir}/models_metadata.pkl"
    with open(target_model_metadata_path, "rb") as f:
        target_model_metadata = joblib.load(f)

    # Get the target model
    target_model_path = f"{log_dir}/model_0.pkl"
    with open(target_model_path, "rb") as f:
        if "adult" in user_configs["data"]["dataset"]:
            target_model = shadow_models.NN(
                configs["train"]["inputs"], configs["train"]["outputs"]
            )  # TODO: read metadata to get the model
        elif "cifar10" in user_configs["data"]["dataset"]:
            target_model = shadow_models.ConvNet()
        target_model.load_state_dict(torch.load(f))

    # Get the population dataset
    data_dir = configs["data"]["data_dir"]
    data_file = configs["data"]["dataset"]
    dataset_path = f"{data_dir}/{data_file}.pkl"
    with open(dataset_path, "rb") as file:
        population = joblib.load(file)
    # ------------------------------------------------
    # Now we have the target model, its metadata, and the train/test dataset
    # indices.
    attack_scheduler = AttackScheduler(
        population,
        target_model,
        target_model_metadata,
        configs,
        logger,
    )
    audit_results = attack_scheduler.run_attacks()

    report_log = configs["audit"]["report_log"]
    privacy_game = configs["audit"]["privacy_game"]
    n_shadow_models = configs["audit"]["num_shadow_models"]
    n_attack_data_size = configs["audit"]["f_attack_data_size"]

    prepare_priavcy_risk_report(
            log_dir,
            [audit_results["qmia"]["result_object"]],
            configs["audit"],
            save_path=f"{log_dir}/{report_log}/{privacy_game}/ns_{n_shadow_models}_fs_{n_attack_data_size}",
        )
