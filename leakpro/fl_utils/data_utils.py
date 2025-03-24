"""Util functions relating to data."""
from abc import ABC, abstractmethod
from copy import deepcopy
from torch import Tensor, cat, mean, randn, std, tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import numpy as np

class GiaDataModalityExtension(ABC):
    @abstractmethod
    def get_at_data():
        """Get a dataloader mimicing the shape of the original data used for recreating."""
        pass

class GiaImageClassifictaionExtension(GiaDataModalityExtension):

    def get_at_data(self, client_loader: DataLoader) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same labels."""
        img_shape = client_loader.dataset[0][0].shape
        num_images = len(client_loader.dataset)
        reconstruction = randn((num_images, *img_shape))
        labels = []
        for _, label in client_loader:
            labels.extend(label.numpy())
        labels = tensor(labels)
        reconstruction_dataset = TensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)
        return reconstruction, labels, reconstruction_loader
    
class GiaNERExtension(GiaDataModalityExtension):
    
    def get_at_data(client_loader: DataLoader, token_used: np.ndarray=None) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same labels."""
        reconstruction_dataset = copy.deepcopy(client_loader.dataset)
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

class CustomTensorDataset(Dataset):
    def __init__(self, reconstruction: torch.Tensor, labels: list):
        self.reconstruction = reconstruction
        self.labels = labels

    def __len__(self):
        return self.reconstruction.size(0)

    def __getitem__(self, index):
        return self.reconstruction[index], self.labels[index]

class GiaImageDetectionExtension(GiaDataModalityExtension):

    def get_at_data(self, client_loader: DataLoader) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same COCO labels."""
        img_shape = client_loader.dataset[0][0].shape
        num_images = len(client_loader.dataset)
        reconstruction = randn((num_images, *img_shape))
        labels = []
        for _, label in client_loader:
            labels.append(deepcopy(label))
        reconstruction_dataset = CustomTensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)

        return reconstruction, labels, reconstruction_loader

class GiaTextMaskingExtension(GiaDataModalityExtension):
    def get_at_data():
        pass


def get_meanstd(trainset: Dataset, axis_to_reduce: tuple=(-2,-1)) -> tuple[Tensor, Tensor]:
    """Get mean and std of a dataset."""
    cc = cat([trainset[i][0].unsqueeze(0) for i in range(len(trainset))], dim=0)
    axis_to_reduce += (0,)
    data_mean = mean(cc, dim=axis_to_reduce).tolist()
    data_std = std(cc, dim=axis_to_reduce).tolist()
    return data_mean, data_std
