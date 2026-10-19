#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GDD_ENS data handler — UserDataset only (role="data").

Features are used RAW (no normalization or augmentation), so the dataset is a thin wrapper
over the feature/target tensors.

Usage:
    from gdd_data_handler import GddDataHandler
    population = GddDataHandler.UserDataset(features, targets)
"""

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler


class GddDataHandler(AbstractInputHandler, role="data"):
    """Provides UserDataset for GDD_ENS. No training logic."""

    class UserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, data, targets, **kwargs):
            assert data.shape[0] == targets.shape[0], "Data and targets must have the same length"
            self.data = data.float()
            self.targets = targets.long()
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __getitem__(self, index):
            return self.data[index], self.targets[index]

        def __len__(self):
            return len(self.targets)
