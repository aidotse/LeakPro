#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Image batches in data space ([0, 1] pixels) with the statistics the attack's modality needs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

_STATS = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616], 10),
    "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762], 100),
    "celeba": ([0.5063, 0.4258, 0.3832], [0.3107, 0.2904, 0.2897], 2),
}


@dataclass
class ImageBatch:
    """A private client batch: ``images`` in data space ``[N, C, H, W]``, ``labels`` ``[N]``, and the normalisation."""

    images: Tensor
    labels: Tensor
    mean: list[float]
    std: list[float]
    num_classes: int


def image_batch(
    quick: bool,
    num_images: int,
    data_dir: Path,
    device: torch.device,
    *,
    dataset: str = "cifar10",
    start_idx: int = 0,
) -> ImageBatch:
    """``quick``: random 3x16x16 images (a different stream for each ``start_idx``). Otherwise ``num_images`` consecutive images of the dataset (downloaded to ``data_dir``)."""
    if quick:
        # A private generator: the attack seeds the global one, and an image drawn from that same stream would be
        # exactly the attack's uniform starting point.
        # ``start_idx`` offsets the seed, so an auxiliary set (start_idx != 0) is independent of the private batch.
        images = torch.rand(num_images, 3, 16, 16, generator=torch.Generator().manual_seed(7919 + start_idx))
        labels = torch.arange(num_images) % 10
        return ImageBatch(images.to(device), labels.to(device), [0.5] * 3, [0.25] * 3, 10)

    import torchvision  # noqa: PLC0415
    from torchvision import transforms  # noqa: PLC0415

    mean, std, num_classes = _STATS[dataset]
    if dataset == "celeba":
        tf = transforms.Compose([transforms.CenterCrop(178), transforms.Resize(64), transforms.ToTensor()])
        ds = torchvision.datasets.CelebA(root=str(data_dir), split="train", target_type="attr", transform=tf, download=True)
        items = [ds[i] for i in range(start_idx, start_idx + num_images)]
        labels = torch.stack([torch.as_tensor(y[31]).long() for _, y in items])  # Smiling
    else:
        cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
        ds = cls(root=str(data_dir), train=True, download=True, transform=transforms.ToTensor())
        items = [ds[i] for i in range(start_idx, start_idx + num_images)]
        labels = torch.tensor([y for _, y in items])
    images = torch.stack([x for x, _ in items])
    return ImageBatch(images.to(device), labels.to(device), mean, std, num_classes)
