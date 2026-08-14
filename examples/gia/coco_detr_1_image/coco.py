"""COCO detection data loader producing DETR style targets for GIA experiments."""
import os

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor, as_tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from leakpro.fl_utils.data_utils import detr_collate_fn

# DETR (facebook/detr-resnet-50) was trained on ImageNet normalized inputs.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def letterbox_geometry(orig_h: int, orig_w: int, img_size: int) -> tuple[int, int, float, float]:
    """Content size and padding for an aspect preserving fit of an image into a square canvas.

    This reproduces the arithmetic of ``coco_yolo_1_image/coco.py``'s two step ``load_image`` (scale
    the longest side to ``img_size``, truncating) plus ``resize`` (centre the result in the square),
    so that the same COCO image letterboxes to the same geometry in both experiments.
    """
    scale = img_size / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    return new_w, new_h, (img_size - new_w) / 2, (img_size - new_h) / 2


def letterbox_image(image: np.ndarray, img_size: int) -> np.ndarray:
    """Aspect preserving resize of an RGB uint8 image into a square canvas padded with black.

    The padding is applied in raw pixel space, before ImageNet normalization, so the bars denormalize
    back to exactly 0 when the attack's ground truth and reconstruction are written out.
    ``cv2.INTER_LINEAR`` rather than a torchvision resize because the YOLO example resamples that way,
    and PIL's antialiased bilinear differs from it by up to 110/255 on a 2.5x downscale.
    """
    new_w, new_h, pad_w, pad_h = letterbox_geometry(image.shape[0], image.shape[1], img_size)
    resized = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # The +/-0.1 splits an odd number of padding rows between the two sides, as the YOLO example does.
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def letterbox_boxes(boxes: list, orig_h: int, orig_w: int, img_size: int) -> np.ndarray:
    """Map normalized cxcywh boxes from the original image onto the letterboxed canvas.

    Under a plain stretch to a square normalized coordinates are invariant, but letterboxing shifts and
    shrinks the content, so the boxes have to move with it. Boxes are clipped to the canvas exactly as
    the YOLO example's ``xy2wh`` does, which keeps a box that straddles the padding boundary consistent
    between the two pipelines.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(b):
        return b.astype(np.float32)
    new_w, new_h, pad_w, pad_h = letterbox_geometry(orig_h, orig_w, img_size)
    x1 = np.clip(new_w * (b[:, 0] - b[:, 2] / 2) + pad_w, 0, img_size - 1e-3)
    x2 = np.clip(new_w * (b[:, 0] + b[:, 2] / 2) + pad_w, 0, img_size - 1e-3)
    y1 = np.clip(new_h * (b[:, 1] - b[:, 3] / 2) + pad_h, 0, img_size - 1e-3)
    y2 = np.clip(new_h * (b[:, 1] + b[:, 3] / 2) + pad_h, 0, img_size - 1e-3)
    return np.stack([(x1 + x2) / 2 / img_size, (y1 + y2) / 2 / img_size,
                     (x2 - x1) / img_size, (y2 - y1) / img_size], axis=1).astype(np.float32)


class CocoDetrDataset(Dataset):
    """COCO detection dataset returning (image, target) where target is a DETR style dict.

    Images are fitted to a fixed square resolution so that a batch can be stacked into a
    single tensor, which is what the gradient inversion attack optimizes over.
    Boxes are returned as normalized (center_x, center_y, width, height), and class labels use
    the raw COCO category ids, which is the label space of the pretrained DETR checkpoints.

    With ``letterbox`` the fit preserves the aspect ratio and pads with black, matching the YOLO
    example pixel for pixel so the two experiments attack the same image. Without it the image is
    stretched to the square, which is what produced the results tables in ``CLAUDE.md``.
    """

    def __init__(self, root: str, split: str, img_size: int, image_ids: list, letterbox: bool = True) -> None:
        self.image_dir = os.path.join(root, "images", split)
        self.coco = COCO(os.path.join(root, "annotations", f"instances_{split}.json"))
        self.image_ids = image_ids
        self.img_size = img_size
        self.letterbox = letterbox
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.stretch_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, dict]:
        """Get the normalized image and its DETR target dict."""
        image_id = self.image_ids[index]
        info = self.coco.loadImgs(image_id)[0]
        image = Image.open(os.path.join(self.image_dir, info["file_name"])).convert("RGB")
        img_w, img_h = info["width"], info["height"]

        classes, boxes = [], []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)):
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            classes.append(ann["category_id"])
            boxes.append([(x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h])

        if self.letterbox:
            # The decoded shape rather than the annotation's width/height, so the box mapping cannot
            # disagree with the geometry the pixels were actually fitted to.
            pixels = np.asarray(image)
            sample = self.transform(letterbox_image(pixels, self.img_size))
            boxes = letterbox_boxes(boxes, pixels.shape[0], pixels.shape[1], self.img_size)
        else:
            sample = self.stretch_transform(image)

        target = {
            "class_labels": torch.as_tensor(classes, dtype=torch.long),
            "boxes": torch.as_tensor(np.asarray(boxes, dtype=np.float32)).reshape(-1, 4),
        }
        return sample, target


def get_coco_detr_loader(num_images: int = 1, img_size: int = 256, start_idx: int = 0, batch_size: int = 1,
                         num_workers: int = 0, root: str = "COCO", split: str = "val2017",
                         min_objects: int = 1, max_objects: int = 100, image_ids: list = None,
                         letterbox: bool = True
                         ) -> tuple[DataLoader, Tensor, Tensor]:
    """Get a client dataloader with COCO images that have a usable number of annotated objects.

    Passing explicit ``image_ids`` attacks exactly those images and skips the object count filter,
    which is how a specific published image is reproduced. Otherwise images are taken in sorted id
    order from ``start_idx``, keeping those whose object count is in [min_objects, max_objects].

    ``letterbox`` controls how a non square image is fitted to ``img_size``; see ``CocoDetrDataset``.
    """
    if image_ids is not None:
        selected = list(image_ids)
    else:
        coco = COCO(os.path.join(root, "annotations", f"instances_{split}.json"))
        all_ids = sorted(coco.getImgIds())

        selected = []
        for image_id in all_ids[start_idx:]:
            anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=image_id)) if not a.get("iscrowd", 0)]
            if min_objects <= len(anns) <= max_objects:
                selected.append(image_id)
            if len(selected) == num_images:
                break
        if len(selected) < num_images:
            raise ValueError(f"Only found {len(selected)} of {num_images} requested images from index {start_idx}.")

    dataset = CocoDetrDataset(root=root, split=split, img_size=img_size, image_ids=selected,
                              letterbox=letterbox)
    client_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                               num_workers=num_workers, collate_fn=detr_collate_fn)
    data_mean = as_tensor(IMAGENET_MEAN)[:, None, None]
    data_std = as_tensor(IMAGENET_STD)[:, None, None]
    return client_loader, data_mean, data_std
