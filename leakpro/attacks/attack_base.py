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
"""Run optuna to find best hyperparameters."""
import os
from abc import ABC, abstractmethod

from leakpro.utils.import_helper import Self


class AbstractAttack(ABC):
    """Abstract attack template for attack objects."""

    def __init__(self) -> None:
        self.attack_cache_folder_path = "attack_cache_folder"
        os.makedirs(self.attack_cache_folder_path,exist_ok=True)

    def get_configs(self: Self) -> dict:
        """Return configs used for attack."""
        return self.configs

    @abstractmethod
    def run_attack() -> None:
        """Run the attack on the target model and dataset.

        This method is implemented by subclasses (e.g., GIA and MIA attacks),
        each of which provides specific behavior and results.

        Returns
        -------
        Depends on the subclass implementation.

        """
        pass

