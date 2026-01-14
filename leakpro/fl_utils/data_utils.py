"""Util functions relating to data."""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, cat, mean, randn, std
from torch.utils.data import DataLoader, Dataset

from leakpro.utils.import_helper import Any, Self
from copy import deepcopy
from typing import Self, List, Any, Optional, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader

class GiaDataModalityExtension(ABC):
    """Abstract class for data modality extensions for GIA."""

    @abstractmethod
    def get_at_data() -> Tensor:
        """Get a dataloader mimicing the shape of the original data used for recreating."""
        pass

class CustomTensorDataset(Dataset):
    """Custom generic tensor dataset."""

    def __init__(self:Self, reconstruction: torch.Tensor, labels: list) -> None:
        self.reconstruction = reconstruction
        self.labels = labels

    def __len__(self: Self) -> int:
        """Dataset length."""
        return self.reconstruction.size(0)

    def __getitem__(self: Self, index: int) -> tuple[Tensor, Any]:
        """Get item from index."""
        return self.reconstruction[index], self.labels[index]

class CustomYoloTensorDataset(Dataset):
    """Custom generic tensor dataset."""

    def __init__(self:Self, reconstruction: torch.Tensor, labels: list) -> None:
        self.reconstruction = reconstruction
        self.labels = labels

    def __len__(self: Self) -> int:
        """Dataset length."""
        return self.reconstruction.size(0)

    def __getitem__(self: Self, index: int) -> tuple[Tensor, Any]:
        """Get item from index."""
        return self.reconstruction[index], self.labels[index], 1


class GiaImageCloneNoiseExtension(GiaDataModalityExtension):
    """Clone original images and replace a fraction of pixels with random values.

    pixel_noise_p:
      0.0 -> exact copy
      1.0 -> every pixel replaced with random values

    random_mode:
      "normal"  -> randn_like (mean 0, std 1)
      "uniform" -> uniform in [0, 1)
    """
    def __init__(
        self,
        pixel_noise_p: float = 0.0,
        random_mode: Literal["normal", "uniform"] = "normal",
        pixel_wise: bool = True,  # True: mask is (N,1,H,W); False: element-wise (N,C,H,W)
    ):
        super().__init__()
        if not (0.0 <= pixel_noise_p <= 1.0):
            raise ValueError("pixel_noise_p must be in [0, 1]")
        self.pixel_noise_p = float(pixel_noise_p)
        self.random_mode = random_mode
        self.pixel_wise = bool(pixel_wise)

    def _clone_labels(self, loader: DataLoader) -> List[Any]:
        labels: List[Any] = []
        for _, label in loader:
            if isinstance(label, Tensor):
                labels.extend([deepcopy(x) for x in label])
            else:
                labels.append(deepcopy(label))
        return labels

    def _random_like(self, x: Tensor) -> Tensor:
        if self.random_mode == "normal":
            return torch.randn_like(x)
        if self.random_mode == "uniform":
            return torch.rand_like(x)
        raise ValueError(f"Unknown random_mode: {self.random_mode}")

    def get_at_data(self: Self, client_loader: DataLoader):
        original = torch.stack([img.clone() for img, _ in client_loader.dataset], dim=0)
        labels = self._clone_labels(client_loader)

        reconstruction = original.clone()

        p = self.pixel_noise_p
        if p > 0.0:
            # original is expected to be (N,C,H,W)
            if reconstruction.ndim != 4:
                raise ValueError(f"Expected images with shape (N,C,H,W), got {tuple(reconstruction.shape)}")

            n, c, h, w = reconstruction.shape

            if self.pixel_wise:
                # Choose pixels by (H,W), apply to all channels together
                mask = (torch.rand((n, 1, h, w), device=reconstruction.device) < p)
                mask = mask.expand(n, c, h, w)
            else:
                # Element-wise mask, channel elements chosen independently
                mask = (torch.rand((n, c, h, w), device=reconstruction.device) < p)

            rand_vals = self._random_like(reconstruction)

            # Where mask=True, take random; else keep original
            reconstruction = torch.where(mask, rand_vals, reconstruction)

        org_dataset = CustomTensorDataset(original, labels)
        org_loader = DataLoader(org_dataset, batch_size=32, shuffle=True)

        reconstruction_dataset = CustomTensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)

        return org_loader, original, reconstruction, labels, reconstruction_loader



class GiaImageExtension(GiaDataModalityExtension):
    """Image extension for GIA."""

    def get_at_data(self: Self, client_loader: DataLoader) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same COCO labels."""
        original = torch.stack([
            img.clone()
            for img, _ in client_loader.dataset
        ], dim=0)
        org_labels = []
        for _, label in client_loader:
            if isinstance(label, Tensor):
                org_labels.extend(deepcopy(label))
            else:
                org_labels.append(deepcopy(label))

        # 3) re-make your dataset & loader
        org_dataset = CustomTensorDataset(original, org_labels)
        org_loader  = DataLoader(org_dataset, batch_size=32, shuffle=True)
        img_shape = client_loader.dataset[0][0].shape
        num_images = len(client_loader.dataset)
        reconstruction = randn((num_images, *img_shape))
        labels = []
        for _, label in client_loader:
            if isinstance(label, Tensor):
                labels.extend(deepcopy(label))
            else:
                labels.append(deepcopy(label))
        reconstruction_dataset = CustomTensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)


        return org_loader, original, reconstruction, labels, reconstruction_loader

class GiaImageYoloExtension(GiaDataModalityExtension):
    """Image extension for GIA."""

    def get_at_data(self: Self, client_loader: DataLoader) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same COCO labels."""
        img_shape = client_loader.dataset[0][0].shape
        num_images = len(client_loader.dataset)
        reconstruction = randn((num_images, *img_shape))
        labels = []
        for _, label, _ in client_loader:
            if isinstance(label, Tensor):
                labels.extend(deepcopy(label))
            else:
                labels.append(deepcopy(label))
        reconstruction_dataset = CustomYoloTensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)
        original = torch.stack([
            img.clone()
            for img, _, _ in client_loader.dataset
        ], dim=0)
        org_labels = []
        for _, label, _ in client_loader:
            if isinstance(label, Tensor):
                org_labels.extend(deepcopy(label))
            else:
                org_labels.append(deepcopy(label))
        org_dataset = CustomYoloTensorDataset(original, org_labels)
        org_loader = DataLoader(org_dataset, batch_size=32, shuffle=True)
        return org_loader, original, reconstruction, labels, reconstruction_loader


def get_meanstd(trainset: Dataset, axis_to_reduce: tuple=(-2,-1)) -> tuple[Tensor, Tensor]:
    """Get mean and std of a dataset."""
    cc = cat([trainset[i][0].unsqueeze(0) for i in range(len(trainset))], dim=0)
    cc = cc.float()
    axis_to_reduce += (0,)
    data_mean = mean(cc, dim=axis_to_reduce).tolist()
    data_std = std(cc, dim=axis_to_reduce).tolist()
    return data_mean, data_std

def get_used_tokens(model: torch.nn.Module, client_gradient: Tensor) -> np.array:
    """Get used tokens."""

    # searching for layer index corresponding to the embedding layer
    for i, name in enumerate(model.named_parameters()):
        if name[0] == "embedding_layer.weight":
            embedding_layer_idx = i


    upd_embedding = client_gradient[embedding_layer_idx].detach().cpu().numpy()

    diff = np.sum(abs(upd_embedding),0)
    return np.where(diff>0)[0]
