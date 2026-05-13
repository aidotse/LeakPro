"""
LR_los_data_handler.py — UserDataset for LOS/MIMIC-III (Logistic Regression variant).

Upload this file as dataset_handler.py in Step 1 of the LeakPro webapp.
Pair it with data_LR.pkl (shape: 23944 × 7488, binary targets: LOS > 3 days).

IMPORTANT: data_LR.pkl is already StandardScaler-normalized (mean≈0, std≈1).
           This handler does NOT re-normalize — it passes data through as-is.
           Values outside [0, 1] are expected and correct.
"""
import numpy as np
import torch
from leakpro import AbstractInputHandler


class UserDataset(AbstractInputHandler.UserDataset):
    """
    Wraps the pre-normalized LOS LR dataset for LeakPro.

    data:    float32 Tensor (N, 7488) — already StandardScaler-normalized
    targets: float32 Tensor (N,)      — binary labels (0 = LOS ≤ 3 days, 1 = LOS > 3 days)
    """

    def __init__(self, data, targets, **kwargs):
        # Convert numpy → torch if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets.astype(np.float32))

        self.data = data.float()
        self.targets = targets.float().view(-1)

        # Identity stubs so LeakPro subsets inherit them without re-scaling
        self.mean = 0.0
        self.std  = 1.0

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, index):
        # Data is pre-normalized — return directly
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)
