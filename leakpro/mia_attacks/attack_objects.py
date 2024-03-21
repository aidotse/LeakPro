from ..model import Model
from ..dataset import Dataset
from ..model import PytorchModel
from torch.nn import CrossEntropyLoss

import numpy as np


class AttackObjects:
    def __init__(self, population: Dataset, train_test_dataset, target_model: Model, configs: dict):
        self._population = population
        self._population_size = len(population)
        self._target_model = PytorchModel(target_model, CrossEntropyLoss())
        self._train_test_dataset = train_test_dataset
        self._num_shadow_models = configs["audit"]["num_shadow_models"]

        self._audit_dataset = {
            # Assuming train_indices and test_indices are arrays of indices, not the actual data
            "data": np.concatenate(
                (train_test_dataset["train_indices"], train_test_dataset["test_indices"])
            ),
            # in_members will be an array from 0 to the number of training indices - 1
            "in_members": np.arange(len(train_test_dataset["train_indices"])),
            # out_members will start after the last training index and go up to the number of test indices - 1
            "out_members": np.arange(
                len(train_test_dataset["train_indices"]),
                len(train_test_dataset["train_indices"]) + len(train_test_dataset["test_indices"]),
            ),
        }

    @property
    def population(self):
        return self._population

    @property
    def population_size(self):
        return self._population_size

    @property
    def target_model(self):
        return self._target_model

    @property
    def train_test_dataset(self):
        return self._train_test_dataset

    @property
    def audit_dataset(self):
        return self._audit_dataset
