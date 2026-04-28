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
"""Parent class for user inputs."""

from abc import ABC, abstractmethod

from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.import_helper import Any, Dict, Self


class AbstractInputHandler(ABC):
    """Parent class for user inputs."""

    def __init_subclass__(cls: type, **kwargs:dict) -> None:
        """Enforces that all subclasses must define a nested class named 'UserDataset'."""
        super().__init_subclass__(**kwargs)

        # Check if 'UserDataset' is defined in the subclass
        if not hasattr(cls, "UserDataset") or not issubclass(cls.UserDataset, Dataset):
            raise TypeError(f"Class {cls.__name__} must define a nested class named 'UserDataset'.")


    @abstractmethod
    def train(
        self: Self,
        dataloader: DataLoader,
        model: Module,
        criterion: _Loss,
        optimizer: Optimizer
    ) -> TrainingOutput:
        """Procedure to train a model on data from the population."""
        pass

    @abstractmethod
    def eval(
        self: Self,
        dataloader: DataLoader,
        model: Module,
        criterion: _Loss,
        device: str
    ) -> EvalOutput:
        """Procedure to train a model on data from the population."""
        pass

    class UserDataset(Dataset, ABC):
        """Parent class for user-defined datasets."""

        @abstractmethod
        def __init__(self: Self, data: Any, targets: Any, **kwargs: dict) -> None:
            """Abstract base class for datasets. Must be implemented in subclasses."""
            pass

        def return_params(self:Self) -> Dict[str, Any]:
            """Returns required parameters dynamically."""
            return {k: v for k, v in vars(self).items() if k not in {"data", "targets"}}

        def __len__(self: Self) -> int:
            """Return the length of the dataset."""
            return len(self.targets)

        def __getitem__(self: Self, index:int) -> Any:
            """Return a sample from the dataset."""
            return self.data[index], self.targets[index]
