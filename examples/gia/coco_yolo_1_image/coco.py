from copy import deepcopy
import time
from torch import Tensor, as_tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from leakpro.fl_utils.data_utils import get_meanstd
from random import sample


import os
from pathlib import Path
from typing import Optional, Callable, Union, List, Tuple
from PIL import Image
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms

class CocoDetectionWithSeparateTransforms(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # We set transforms=None here because we'll handle image and target transforms separately.
        super().__init__(root, transforms=None, transform=transform, target_transform=target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
    
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    
    def _load_target(self, id: int) -> List[dict]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[dict]]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        
        # Capture the original size before applying the image transform
        original_size = image.size  # (width, height)
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target, original_size)
        
        return image, target
    
    def __len__(self) -> int:
        return len(self.ids)

def resize_target(target: List[dict], original_size: Tuple[int, int], new_size: Tuple[int, int] = (64, 64)) -> List[dict]:
    orig_w, orig_h = original_size
    new_w, new_h = new_size
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    new_target = []
    for obj in target:
        new_obj = deepcopy(obj)
        if 'bbox' in new_obj:
            x, y, w, h = new_obj['bbox']
            new_obj['bbox'] = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]
        new_target.append(new_obj)
    return new_target

def custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

def get_coco_detection_loader(num_images: int = 1, img_size=256, start_idx=0, batch_size: int = 1, num_workers: int = 2, root: str = './coco2017', ann_file: str = 'annotations/instances_train2017.json') -> tuple[DataLoader, Tensor, Tensor]:
    """Get a dataloader for COCO detection with non-empty labels."""
    ann_file = f"{root}/annotations/instances_train2017.json"
    img_dir = f"{root}/train2017"
    dataset = CocoDetectionWithSeparateTransforms(
        root=img_dir, 
        annFile=ann_file, 
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)), 
            transforms.ToTensor()
        ]), 
        target_transform=lambda target, orig_size: resize_target(target, orig_size, new_size=(img_size, img_size))
    )
    
    print("warning low mean std calcualting currently :)")
    # Compute data_mean and data_std on a small random subset of the dataset.
    subset_indices = sample(range(len(dataset)), min(len(dataset), 100))
    data_mean, data_std = get_meanstd(Subset(dataset, subset_indices))
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)
    ])

    total_examples = len(dataset)
    filtered_indices = []
    current_idx = start_idx

    # Iterate until we have num_images with non-empty labels or we run out of examples.
    while len(filtered_indices) < num_images and current_idx < total_examples:
        # Assuming dataset[current_idx] returns (image, target)
        _, target = dataset[current_idx]
        if len(target) > 0:
            filtered_indices.append(current_idx)
        current_idx += 1
    if len(filtered_indices) < num_images:
        print(f"Warning: Only found {len(filtered_indices)} images with non-empty labels out of requested {num_images}")

    subset_trainset = Subset(dataset, filtered_indices)
    subset_trainset.dataset.transform = transform

    client_loader = DataLoader(
        subset_trainset, 
        batch_size=batch_size, 
        collate_fn=custom_collate_fn,
        shuffle=False, 
        drop_last=True, 
        num_workers=num_workers
    )
    data_mean = as_tensor(data_mean)[:, None, None]
    data_std = as_tensor(data_std)[:, None, None]
    return client_loader, data_mean, data_std

def batch_targets_to_tensor(targets: Tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_list = []
    for img_idx, image_annotations in enumerate(targets):
        # Each image_annotations is a list of dictionaries.
        for annotation in image_annotations:
            # Assuming annotation["bbox"] is in [x, y, w, h] format.
            bbox = torch.tensor(annotation["bbox"], dtype=torch.float32, device=device)
            # Convert category id to a tensor.
            # We use float here so that the whole tensor has a single dtype.
            # You can convert it to long later if needed.
            category_id = torch.tensor([annotation["category_id"] - 1], dtype=torch.float32, device=device)
            # Prepend the image index.
            image_idx_tensor = torch.tensor([img_idx], dtype=torch.float32, device=device)
            # Concatenate image index, category id, and bbox.
            target_tensor = torch.cat((image_idx_tensor, category_id, bbox))
            gt_list.append(target_tensor)
    # Stack all the target tensors into a single tensor.
    targets = torch.stack(gt_list)
    return targets