import time
from torch import Tensor, as_tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CocoDetection
from leakpro.fl_utils.data_utils import get_meanstd
from random import sample

def get_coco_detection_loader(num_images: int = 1, batch_size: int = 1, num_workers: int = 2, root: str = './coco2017', ann_file: str = 'annotations/instances_train2017.json') -> tuple[DataLoader, Tensor, Tensor]:
    """Get a dataloader for COCO detection."""
    ann_file = f"{root}/annotations/instances_train2017.json"
    img_dir = f"{root}/train2017"
    dataset = CocoDetection(root=img_dir, annFile=ann_file, transform=transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()]))
    subset_indices = sample(range(len(dataset)), min(len(dataset), 200))
    data_mean, data_std = get_meanstd(Subset(dataset, subset_indices))
    transform = transforms.Compose([
            transforms.Normalize(data_mean, data_std)])

    total_examples = len(dataset)
    end_idx = min(17 + num_images, total_examples)
    indices = list(range(17, end_idx))
    subset_trainset = Subset(dataset, indices)
    subset_trainset.dataset.transform = transform
    trainloader = DataLoader(subset_trainset, batch_size=batch_size,
                                            shuffle=False, drop_last=True, num_workers=num_workers)
    data_mean = as_tensor(data_mean)[:, None, None]
    data_std = as_tensor(data_std)[:, None, None]
    return trainloader, data_mean, data_std
