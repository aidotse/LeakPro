from .dataset import Dataset
from typing import List, Tuple
from torch import nn
from .information_source import InformationSource
from .dataset import Dataset, get_dataset_subset
from .signal import ModelLoss, ModelNegativeRescaledLogits
from .hypothesis_test import linear_itp_threshold_func
from .model import PytorchModel
from .metric import PopulationMetric
from . import audit_report
from .audit_report import ROCCurveReport, SignalHistogramReport, VulnerablePointsReport
from .constants import InferenceGame

from typing import List
from torch import nn
from pathlib import Path

def prepare_information_source(
    log_dir: str,
    dataset: Dataset,
    data_split: dict,
    model_list: List[nn.Module],
    configs: dict,
    model_metadata_dict: dict,
    target_model_idx_list: List[int] = None,
    model_name: str = None,
    dataset_name: str = None,
):
    """Prepare the information source for calling the core of the Privacy Meter
    Args:
        log_dir (str): Log directory that saved all the information, including the models.
        dataset (torchvision.datasets): The whole dataset
        data_split (dict): Data split information. 'split' contains a list of dict, each of which has the train, test and audit information. 'split_method' indicates the how the dataset is generated.
        model_list (List): List of target models.
        configs (dict): Auditing configuration.
        model_metadata_dict (dict): Model metedata dict.
        model_name str: target model name
        dataset_name (str): name of the dataset

    Returns:
        List(InformationSource): target information source list.
        List(InformationSource): reference information source list.
        List: List of metrics used for each target models.
        List(str): List of directory to save the Privacy Meter results for each target model.
        dict: Updated metadata for the trained model.
    """
    reference_info_source_list = []
    target_info_source_list = []
    metric_list = []
    log_dir_list = []

    # Prepare the information source for each target model
    split = 0
    print(f"preparing information sources for target model")
    log_dir_path = f"{log_dir}/{configs['report_log']}/signal_{split}"
    signals, hypothesis_test_func = get_signal_and_hypothesis_test_func(configs)

    (
        target_dataset,
        audit_dataset,
        target_model,
        audit_models,
    ) = get_info_source_population_attack(
        dataset,
        data_split["split"][split],
        model_list[split],
        configs,
        model_name,
    )
    target_info_source = InformationSource(
        models=target_model, datasets=target_dataset
    )
    reference_info_source = InformationSource(
        models=audit_models, datasets=audit_dataset
    )
    metrics = PopulationMetric(
        target_info_source=target_info_source,
        reference_info_source=reference_info_source,
        signals=signals,
        hypothesis_test_func=hypothesis_test_func,
        logs_dirname=log_dir_path,
    )

    metric_list.append(metrics)

    reference_info_source = InformationSource(
        models=audit_models, datasets=audit_dataset
    )
    reference_info_source_list.append(reference_info_source)
    target_info_source_list.append(target_info_source)

    # Save the log_dir for attacking different target model
    Path(log_dir_path).mkdir(parents=True, exist_ok=True)
    log_dir_list.append(log_dir_path)

    return (
        target_info_source_list,
        reference_info_source_list,
        metric_list,
        log_dir_list,
        model_metadata_dict,
    )
    

def get_signal_and_hypothesis_test_func(configs):
    """Return the attack and way to find the threshold

    Args:
        configs (dict): Auditing configuration.
    """
    signals = []

    if configs["signal"] == "loss":
        signals.append(ModelLoss())
    elif configs["signal"] == "rescaled_logits":
        signals.append(ModelNegativeRescaledLogits())
    else:
        raise ValueError(
            f"{configs['signal']} is not supported. Please use loss or rescaled_logits as the signal."
        )

    hypothesis_test_func = linear_itp_threshold_func
    return signals, hypothesis_test_func


def get_info_source_population_attack(
    dataset: Dataset,
    data_split: dict,
    model: nn.Module,
    configs: dict,
    model_name: str,
):
    """Prepare the information source for calling the core of Privacy Meter for the population attack

    Args:
        dataset(torchvision.datasets): The whole dataset
        data_split (dict): Data split information. 'split' contains a list of dict, each of which has the train, test and audit information. 'split_method' indicates the how the dataset is generated.
        model (nn.Module): Target Model.
        configs (dict): Auditing configuration
        model_name (str): Target model name
    Returns:
        List(Dataset): List of target dataset on which we want to infer the membership
        List(Dataset):  List of auditing datasets we use for launch the attack
        List(nn.Module): List of target models we want to audit
        List(nn.Module): List of reference models (which is the target model based on population attack)
    """
    train_data, train_targets = get_dataset_subset(
        dataset, data_split["train"], model_name, device=configs["device"]
    )
    test_data, test_targets = get_dataset_subset(
        dataset, data_split["test"], model_name, device=configs["device"]
    )
    audit_data, audit_targets = get_dataset_subset(
        dataset, data_split["audit"], model_name, device=configs["device"]
    )
    target_dataset = Dataset(
        data_dict={
            "train": {"x": train_data, "y": train_targets},
            "test": {"x": test_data, "y": test_targets},
        },
        default_input="x",
        default_output="y",
    )

    audit_dataset = Dataset(
        data_dict={"train": {"x": audit_data, "y": audit_targets}},
        default_input="x",
        default_output="y",
    )
    target_model = PytorchModel(
        model_obj=model,
        loss_fn=nn.CrossEntropyLoss()
    )
    return [target_dataset], [audit_dataset], [target_model], [target_model]



def prepare_priavcy_risk_report(
    log_dir: str, audit_results: List, configs: dict, save_path: str = None, target_info_source: InformationSource = None, target_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None
):
    """Generate privacy risk report based on the auditing report

    Args:
        log_dir(str): Log directory that saved all the information, including the models.
        audit_results(List): Privacy meter results.
        configs (dict): Auditing configuration.
        save_path (str, optional): Report path. Defaults to None.

    Raises:
        NotImplementedError: Check if the report for the privacy game is implemented.

    """
    audit_report.REPORT_FILES_DIR = "privacy_meter/report_files"
    if save_path is None:
        save_path = log_dir

    if configs["privacy_game"] in ["privacy_loss_model","avg_privacy_loss_training_algo"]:
        # Generate privacy risk report for auditing the model
        if len(audit_results) == 1 and configs["privacy_game"] == "privacy_loss_model":
            ROCCurveReport.generate_report(
                metric_result=audit_results[0],
                inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
                save=True,
                filename=f"{save_path}/ROC.png",
            )
            SignalHistogramReport.generate_report(
                metric_result=audit_results[0][0],
                inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
                save=True,
                filename=f"{save_path}/Histogram.png",
            )
            # VulnerablePointsReport.generate_report(
            #     metric_results=audit_results[0],
            #     inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
            #     target_info_source=target_info_source,
            #     target_model_to_train_split_mapping=target_model_to_train_split_mapping,
            #     filename = f"{save_path}/VulnerablePoints.png",
            # )
        else:
            raise ValueError(
                f"{len(audit_results)} results are not enough for {configs['privacy_game']})"
            )
    else:
        raise NotImplementedError(f"{configs['privacy_game']} is not implemented yet")
