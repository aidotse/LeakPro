import logging
import argparse
import os
import random
import numpy as np
import torch
import yaml
import pickle
from pathlib import Path
import time
import leakpro.dataset as dataset
import leakpro.models as models
import leakpro.train as util
#from leakpro.core import prepare_information_source, prepare_priavcy_risk_report
#from leakpro.audit import Audit
from leakpro.mia_attacks.attack_scheduler import AttackScheduler

def setup_log(name: str, save_file: bool):
    """Generate the logger for the current run.
    Args:
        name (str): Logging file name.
        save_file (bool): Flag about whether to save to file.
    Returns:
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
    
    args = "./config/adult.yaml"
    with open(args, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)
     
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
    
    #------------------------------------------------
    # Create the population dataset
    population = dataset.get_dataset(configs["data"]["dataset"], configs["data"]["data_dir"])
    N_population = len(population)
    
    # Create target training dataset and test dataset 
    # NOTE: this should not be done as the model is provided by the user
    train_test_dataset = dataset.prepare_train_test_datasets(N_population, configs["data"])
    
    train_loader = dataset.get_dataloader(torch.utils.data.Subset(population, 
                                                                  train_test_dataset["train_indices"]),
                                          batch_size=configs["train"]["batch_size"],
                                          shuffle=True,)
    test_loader = dataset.get_dataloader(torch.utils.data.Subset(population, 
                                                                 train_test_dataset["test_indices"]),
                                         batch_size=configs["train"]["test_batch_size"],
                                         )
    
    # Train the target model
    model = models.NN(configs["train"]["inputs"], configs["train"]["outputs"])
    model = util.train(model, train_loader, configs, test_loader, train_test_dataset)
    
    #------------------------------------------------
    # LEAKPRO starts here
    # Read in model, population, and metadata    
    dataset_path = f"{configs["data"]["data_dir"]}/{configs["data"]["dataset"]}.pkl"
    with open(dataset_path, "rb") as file:
        population = pickle.load(file)
    
    
    # Get the training and test data
    train_test_data = train_test_dataset
    
    # Get the target model + metadata
    target_model_metadata_path = f"{log_dir}/models_metadata.pkl"
    with open(target_model_metadata_path, "rb") as f:
        target_model_metadata = pickle.load(f)
    target_model_path = f"{log_dir}/model_0.pkl"
    with open(target_model_path, "rb") as f:
        target_model = models.NN(configs["train"]["inputs"], configs["train"]["outputs"]) # TODO: read metadata to get the model
        target_model.load_state_dict(torch.load(f))
    
    #------------------------------------------------
    # Now we have the target model, its metadata, and the train/test dataset indices.
    attack_scheduler = AttackScheduler(population, train_test_dataset, target_model, target_model_metadata, configs, log_dir,logger) #TODO metadata includes indices for train and test data
    audit_results = attack_scheduler.run_attacks()
    
    logger.info(str(audit_results["attack_p"]["result_object"]))
    
    
    # prepare_priavcy_risk_report(
    #         log_dir,
    #         audit_results,
    #         configs["audit"],
    #         save_path=f"{log_dir}/{configs['audit']['report_log']}",
    #         target_info_source=target_info_source,
    #         target_model_to_train_split_mapping = data_split_info["split"][0]["train"]
    #     )