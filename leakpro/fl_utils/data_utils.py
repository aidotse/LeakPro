"""Util functions relating to data."""
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np

from torch.utils.data import DataLoader, Dataset
from data.data import pre_process_data, Dataset, LabelSet, TrainingBatch

import torch
from torch import Tensor, cat, mean, randn, std

from leakpro.utils.import_helper import Any, Self
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
        color = torch.tensor([150, 150, 30], dtype=original.dtype, device=original.device) / 255.
        color = color.view(1, 3, 1, 1)

        # 2) expand to (N,3,H,W)
        N, C, H, W = original.shape
        constant_images = color.expand(N, C, H, W).clone()

        # 3) re-make your dataset & loader
        org_dataset = CustomTensorDataset(constant_images, org_labels)
        org_loader  = DataLoader(org_dataset, batch_size=32, shuffle=True)
        # org_dataset = CustomTensorDataset(original, org_labels)
        # org_loader = DataLoader(org_dataset, batch_size=32, shuffle=True)
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


        return org_loader, constant_images, reconstruction, labels, reconstruction_loader

class GiaImageYoloExtension(GiaDataModalityExtension):
    """Image extension for GIA."""

    def get_at_data(self: Self, client_loader: DataLoader) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same COCO labels."""
        img_shape = client_loader.dataset[0][0].shape
        num_images = len(client_loader.dataset)
        reconstruction = randn((num_images, *img_shape))
        # reconstruction = torch.stack([
        #     img.clone()
        #     for img, _, _ in client_loader.dataset
        # ], dim=0)
        # o = 1.5
        # noise = torch.randn_like(reconstruction) * o
        # reconstruction = (reconstruction + noise).clamp(0.0, 1.0)
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


class GiaNERExtension(GiaDataModalityExtension):
    
    def get_at_data(self,client_loader: DataLoader, token_used: np.ndarray=None) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same labels."""
        reconstruction_dataset = deepcopy(client_loader.dataset)
        reconstruction = []  # Collect embeddings to optimize
        
        for d in reconstruction_dataset:
            
            ind = np.where(np.array(d.labels)!=0)[0]#[0:1]
            
            token_used = torch.tensor(token_used, device=d.embedding.device)
            ind = torch.tensor(ind, device=d.embedding.device)

            d.embedding.index_put_((ind[:, None], token_used), torch.ones_like(d.embedding[ind[:, None],token_used])/54)#54)


            #d.embedding[ind] = (d.embedding[ind].T/torch.sum(d.embedding[ind],1)).T
            d.embedding.requires_grad = True 
            
            # make tokens with labels != 0 trainable
            mask = torch.zeros_like(d.embedding)
            mask[ind] = 1 
            #d.embedding = PartialTrainableTensor.apply(d.embedding, mask).detach().requires_grad_(True)
            def mask_grad(grad):
                return grad * mask  # Apply the mask to the gradient
            
            d.embedding.register_hook(mask_grad)

            # Add reference to the embedding for optimization
            reconstruction.append(d.embedding)

        reconstruction_loader = DataLoader(reconstruction_dataset, collate_fn=TrainingBatch, batch_size=1, shuffle=False)
        return reconstruction, reconstruction_loader





def get_meanstd(trainset: Dataset, axis_to_reduce: tuple=(-2,-1)) -> tuple[Tensor, Tensor]:
    """Get mean and std of a dataset."""
    cc = cat([trainset[i][0].unsqueeze(0) for i in range(len(trainset))], dim=0)
    cc = cc.float()
    axis_to_reduce += (0,)
    data_mean = mean(cc, dim=axis_to_reduce).tolist()
    data_std = std(cc, dim=axis_to_reduce).tolist()
    return data_mean, data_std

def get_used_tokens(model, client_gradient) -> np.array:

        # searching for layer index corresponding to the embedding layer
        for i, name in enumerate(model.named_parameters()):
            if name[0] == "embedding_layer.weight":
                embedding_layer_idx = i

        
        upd_embedding = client_gradient[embedding_layer_idx].detach().cpu().numpy()
        
        diff = np.sum(abs(upd_embedding),0)
        token_used = np.where(diff>0)[0]
        return token_used