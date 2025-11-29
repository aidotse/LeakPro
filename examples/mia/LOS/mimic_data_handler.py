import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor, unique, from_numpy
from leakpro import AbstractInputHandler



class MIMICUserDataset(AbstractInputHandler.UserDataset):
    """
    A custom dataset class for handling user data.

    Args:
        x (torch.Tensor): The input features as a torch tensor.
        y (torch.Tensor): The target labels as a torch tensor.
        
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.
        subset(indices): Returns a subset of the dataset based on the given indices.
    """

    def __init__(self, data, targets, **kwargs):
        """
        Args:
            data (torch.Tensor): The input features as a torch tensor.
            targets (torch.Tensor): The target labels as a torch tensor.
            mean (Tensor, optional): Precomputed mean for normalization.
            std (Tensor, optional): Precomputed std for normalization.
        """
        assert data.shape[0] == targets.shape[0], "Mismatch between number of samples in data and targets"
        assert set(unique(targets.int()).tolist()).issubset({0, 1}), "Target labels should be either 0 or 1"

        # Ensure both x and y are converted to tensors (float32 type)
        self.data = Tensor(data).float()
        self.targets = Tensor(targets).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y.squeeze(0)
    
def to_3D_tensor(df):
    idx = pd.IndexSlice
    np_3D = np.dstack([df.loc[idx[:, :, :, i], :].values for i in sorted(set(df.index.get_level_values("hours_in")))])
    return from_numpy(np_3D)

