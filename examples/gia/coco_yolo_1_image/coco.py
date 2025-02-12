from torch import Tensor, as_tensor, randperm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CocoDetection

from leakpro.fl_utils.data_utils import get_meanstd


def get_coco_detection_loader(num_images: int = 1, batch_size: int = 1, num_workers: int = 2, root: str = './coco2017', ann_file: str = 'annotations/instances_train2017.json') -> tuple[DataLoader, Tensor, Tensor]:
    """Get a dataloader for COCO detection."""
    ann_file = f"{root}/annotations/instances_train2017.json"
    img_dir = f"{root}/train2017"
    trainset = CocoDetection(root=img_dir, annFile=ann_file, transform=transforms.ToTensor())
    data_mean, data_std = get_meanstd(trainset)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])
    trainset.transform = transform

    total_examples = len(trainset)
    random_indices = randperm(total_examples)[:num_images]
    subset_trainset = Subset(trainset, random_indices)
    trainloader = DataLoader(subset_trainset, batch_size=batch_size,
                                            shuffle=False, drop_last=True, num_workers=num_workers)
    data_mean = as_tensor(data_mean)[:, None, None]
    data_std = as_tensor(data_std)[:, None, None]
    return trainloader, data_mean, data_std
