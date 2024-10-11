"""Util functions relating to data."""
from torch import Tensor, cat, mean, randn, std, tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


def get_meanstd(trainset: Dataset) -> tuple[Tensor, Tensor]:
    """Get mean and std of a dataset."""
    cc = cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = mean(cc, dim=1).tolist()
    data_std = std(cc, dim=1).tolist()
    return data_mean, data_std

def get_at_images(client_loader: DataLoader) -> DataLoader:
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
    return reconstruction, reconstruction_loader
