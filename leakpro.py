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
from leakpro.core import prepare_information_source, prepare_priavcy_risk_report
from leakpro.audit import Audit

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
    if save_file:
        log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
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
    
    # Load the data
    data = dataset.get_dataset(configs["data"]["dataset"], configs["data"]["data_dir"])
    data_split_info = dataset.prepare_datasets(len(data), configs["data"])
    
    train_loader = dataset.get_dataloader(torch.utils.data.Subset(data, data_split_info["split"][0]["train"]),batch_size=configs["train"]["batch_size"],shuffle=True,)
    test_loader = dataset.get_dataloader(torch.utils.data.Subset(data, data_split_info["split"][0]["test"]),batch_size=configs["train"]["test_batch_size"],)
    
    # Load the model
    if os.path.exists((f"{log_dir}/models_metadata.pkl")):
        with open(f"{log_dir}/models_metadata.pkl", "rb") as f:
            model_metadata_list = pickle.load(f)
        with open(f"{log_dir}/model_0.pkl", "rb") as f:
            model = models.NN(configs["train"]["inputs"], configs["train"]["outputs"])
            model.load_state_dict(torch.load(f))
    else:
        model = models.NN(configs["train"]["inputs"], configs["train"]["outputs"])
        model = util.train(model,train_loader,configs,test_loader,data_split_info)
        with open(f"{log_dir}/models_metadata.pkl", "rb") as f:
            model_metadata_list = pickle.load(f)
    
    
    # Perform the auditing
    (
        target_info_source,
        reference_info_source,
        metrics,
        log_dir_list,
        model_metadata_list,
        ) = prepare_information_source(
            log_dir,
            data,
            data_split_info,
            [model],
            configs["audit"],
            model_metadata_list,
            [0],
            configs["train"]["model_name"],
            configs["data"]["dataset"],
        )
    
    inference_game_type = configs["audit"]["privacy_game"].upper()
    audit_obj = Audit(
            metrics=metrics,
            inference_game_type=inference_game_type,
            target_info_sources=target_info_source,
            reference_info_sources=reference_info_source,
            fpr_tolerances=None,
            logs_directory_names=log_dir_list,
        )
    audit_obj.prepare()
    audit_results = audit_obj.run()
    
    prepare_priavcy_risk_report(
            log_dir,
            audit_results,
            configs["audit"],
            save_path=f"{log_dir}/{configs['audit']['report_log']}",
            target_info_source=target_info_source,
            target_model_to_train_split_mapping = data_split_info["split"][0]["train"]
        )