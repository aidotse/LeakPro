"""General util functions."""
import json
import os
from typing import Tuple

# Default path to save results
DEFAULT_PATH_RESULTS = os.path.dirname(__file__) + "/results/"

def aux_file_path(*, prefix: str, dataset: str, path: str = None) -> Tuple[str,str]:
    """Util function that returns the file path for given prefix and dataset and path."""
    if prefix:
        prefix += "_"
    file = "res_" + prefix + dataset + ".json"
    if path is not None:
        if path[-1] != "/":
            path += "/"
    else:
        path = DEFAULT_PATH_RESULTS
    return path + file

def save_res_json_file(*, prefix: str, dataset: str, res: dict, path: str = None) -> None:
    """Util function that saves results dictionary into a json file under given path with given prefix and dataset name."""
    file_path = aux_file_path(prefix=prefix, dataset=dataset, path=path)
    #Create directory if does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(res, f, indent=4)
    print("\n### Results saved!", os.path.basename(file_path)) # noqa: T201

def load_res_json_file(*, prefix: str, dataset: str, path: str = None) -> dict:
    """Util function that loads and returns results from json file under given path with given prefix and dataset name."""
    file_path = aux_file_path(prefix=prefix, dataset=dataset, path=path)
    with open(file_path, "r") as f:
        return json.load(f)
