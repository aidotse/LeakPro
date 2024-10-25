"""General util functions."""
import json
import os
from typing import Tuple

#Path to save results
PATH_RESULTS = os.path.dirname(os.path.dirname(__file__)) + "/synthetic_data_attacks/results/"

def aux_file_path(*, prefix: str, dataset: str) -> Tuple[str,str]:
    """Util function that returns file and file_path for given prefix and dataset."""
    if prefix:
        prefix += "_"
    file = "res_" + prefix + dataset + ".json"
    file_path = PATH_RESULTS + file
    return file, file_path

def save_res_json_file(*, prefix: str, dataset: str, res: dict) -> None:
    """Util function that saves results dictionary into a json file with given prefix and dataset name."""
    file, file_path = aux_file_path(prefix=prefix, dataset=dataset)
    #Create directory if does not exist
    os.makedirs(os.path.dirname(PATH_RESULTS), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(res, f, indent=4)
    print("\n### Results saved!", file) # noqa: T201

def load_res_json_file(*, prefix: str, dataset: str) -> dict:
    """Util function that loads and returns results from json file with given prefix and dataset name."""
    _, file_path = aux_file_path(prefix=prefix, dataset=dataset)
    with open(file_path, "r") as f:
        return json.load(f)
