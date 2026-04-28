#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
