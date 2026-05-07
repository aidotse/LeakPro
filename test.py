import os
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

# Always store data under client/data
OUT_DIR = os.path.join(abs_path, "data")


# --------- RNG utilities (seed then unseed) ---------

@dataclass
class RNGSnapshot:
    torch_state: torch.ByteTensor
    torch_cuda_states: Optional[List[torch.ByteTensor]]
    np_state: tuple
    py_state: object  # random.getstate() if you also want Python random


def seed_everything(seed: int) -> RNGSnapshot:
    """Seed torch + numpy; return previous RNG states so we can restore them."""
    import random

    snap = RNGSnapshot(
        torch_state=torch.get_rng_state(),
        torch_cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        np_state=np.random.get_state(),
        py_state=random.getstate(),
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return snap


def restore_rng(snapshot: RNGSnapshot) -> None:
    """Restore RNG states (unseed)."""
    import random

    random.setstate(snapshot.py_state)
    np.random.set_state(snapshot.np_state)
    torch.set_rng_state(snapshot.torch_state)
    if torch.cuda.is_available() and snapshot.torch_cuda_states is not None:
        torch.cuda.set_rng_state_all(snapshot.torch_cuda_states)


# --------- Dataset prep ---------

def prepare_data(out_dir: str = OUT_DIR) -> None:
    """Download Imagenette once."""
    os.makedirs(out_dir, exist_ok=True)

    # Imagenette uses ImageFolder-style structure after download.
    # Torchvision will download & prepare under root/imagenette2-*
    _ = torchvision.datasets.Imagenette(root=out_dir, split="train", download=True)
    _ = torchvision.datasets.Imagenette(root=out_dir, split="val", download=True)


def _default_transforms(train: bool = True):
    # ConvNeXt expects ImageNet normalization
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


def load_imagenette(
    out_dir: str = OUT_DIR,
    train: bool = True,
):
    split = "train" if train else "val"
    tfm = _default_transforms(train=train)
    return torchvision.datasets.Imagenette(root=out_dir, split=split, transform=tfm, download=False)


# --------- Dirichlet partitioning ---------

def _get_targets(dataset) -> np.ndarray:
    # Imagenette returns (PIL, target). For torchvision datasets, targets are typically in dataset.targets
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets, dtype=np.int64)
    # Fallback: iterate (slower)
    return np.array([dataset[i][1] for i in range(len(dataset))], dtype=np.int64)


from typing import List, Tuple
import numpy as np


def _build_class_proportions(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
) -> np.ndarray:
    """
    Sample Dirichlet client proportions per class from `targets`.

    Returns:
        proportions_per_class: shape [n_classes, num_clients]
    """
    n_classes = int(targets.max() + 1)
    proportions_per_class = np.zeros((n_classes, num_clients), dtype=np.float64)

    for c in range(n_classes):
        proportions_per_class[c] = np.random.dirichlet(
            alpha=np.full(num_clients, alpha, dtype=np.float64)
        )

    return proportions_per_class


def _apply_class_proportions(
    targets: np.ndarray,
    proportions_per_class: np.ndarray,
) -> List[np.ndarray]:
    """
    Partition indices in `targets` according to provided per-class client proportions.

    Args:
        targets: Label array for this split.
        proportions_per_class: shape [n_classes, num_clients]

    Returns:
        List of index arrays, one per client.
    """
    n_classes, num_clients = proportions_per_class.shape
    all_indices = np.arange(len(targets))

    class_indices = [all_indices[targets == c] for c in range(n_classes)]
    for c in range(n_classes):
        np.random.shuffle(class_indices[c])

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(n_classes):
        cls_idx = class_indices[c]
        n_c = len(cls_idx)

        if n_c == 0:
            continue

        proportions = proportions_per_class[c]
        counts = (proportions * n_c).astype(int)

        # Fix rounding so sum(counts) == n_c
        diff = n_c - counts.sum()
        if diff > 0:
            for idx in np.argsort(-proportions)[:diff]:
                counts[idx] += 1
        elif diff < 0:
            for idx in np.argsort(-counts)[:(-diff)]:
                if counts[idx] > 0:
                    counts[idx] -= 1

        start = 0
        for client_id in range(num_clients):
            cnt = int(counts[client_id])
            if cnt > 0:
                sl = cls_idx[start:start + cnt]
                client_indices[client_id].extend(sl.tolist())
                start += cnt

    result = []
    for client_id in range(num_clients):
        arr = np.array(client_indices[client_id], dtype=np.int64)
        np.random.shuffle(arr)
        result.append(arr)

    return result


def dirichlet_partition_indices(
    targets_train: np.ndarray,
    targets_val: np.ndarray,
    num_clients: int,
    alpha: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Partition both train and val indices so that val follows the same
    class/client distribution pattern as train.

    The Dirichlet proportions are sampled once from the train setup and then
    reused for val.

    Returns:
        train_partitions, val_partitions
    """
    n_classes_train = int(targets_train.max() + 1)
    n_classes_val = int(targets_val.max() + 1)
    if n_classes_train != n_classes_val:
        raise ValueError(
            f"Train/val class mismatch: train has {n_classes_train}, "
            f"val has {n_classes_val}"
        )

    proportions_per_class = _build_class_proportions(
        targets=targets_train,
        num_clients=num_clients,
        alpha=alpha,
    )

    train_partitions = _apply_class_proportions(
        targets=targets_train,
        proportions_per_class=proportions_per_class,
    )
    val_partitions = _apply_class_proportions(
        targets=targets_val,
        proportions_per_class=proportions_per_class,
    )

    return train_partitions, val_partitions


def get_client_partition_indices(
    dataset_train,
    dataset_val,
    settings: Dict,
    client_id: str,
    alpha: float = 0.3,
    data_fraction: float = 0.05,
) -> np.ndarray:
    """
    Uses settings['seed'], settings['num_clients'], settings['partition'] (client_id->partition_idx)
    to produce indices for THIS client.
    """
    if settings is None:
        raise ValueError("settings must be provided (needs seed/num_clients/partition).")

    if "seed" not in settings or "num_clients" not in settings or "partition" not in settings:
        raise ValueError("settings must contain keys: 'seed', 'num_clients', 'partition'.")

    seed = int(settings["seed"])
    num_clients = int(settings["num_clients"])
    part_map = settings["partition"]

    if client_id not in part_map:
        raise KeyError(f"client_id '{client_id}' not found in settings['partition'].")
    partition_idx = int(part_map[client_id])

    # Deterministic per-round partitioning:
    # Use a single seed to generate ALL client partitions consistently,
    # then select this client's partition_idx.
    snapshot = seed_everything(seed)
    try:
        # Build deterministic subset of train
        full_train_indices = np.arange(len(dataset_train))
        np.random.shuffle(full_train_indices)
        n_train = max(1, int(len(full_train_indices) * data_fraction))
        subset_train_indices = full_train_indices[:n_train]
        targets_train = _get_targets(dataset_train)[subset_train_indices]
        targets_val = _get_targets(dataset_val)
        parts_train, parts_val = dirichlet_partition_indices(targets_train, targets_val, num_clients=num_clients, alpha=alpha)
        print(f"Returning dirichlet on seed {seed} for partition {partition_idx} and num_clients {num_clients}, len train: {len(parts_train)}")
        return parts_train[partition_idx], parts_val[partition_idx]
    finally:
        # Unseed (restore RNG)
        restore_rng(snapshot)


def make_loaders_for_client(
    settings: Dict,
    client_id: str,
    batch_size: int = 64,
    num_workers: int = 2,
    alpha: float = 0.3,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Returns: train_loader, val_loader, num_train_examples_for_this_client
    """
    train_ds = load_imagenette(train=True)
    val_ds = load_imagenette(train=False)

    idx_train, idx_val = get_client_partition_indices(train_ds, val_ds, settings=settings, client_id=client_id, alpha=alpha)
    train_subset = Subset(train_ds, idx_train.tolist())
    val_subset = Subset(val_ds, idx_val.tolist())
    print(f"len train {len(train_subset)}")
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, len(train_subset)


import time
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    print(f"[START] {label}...")
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"[DONE]  {label} — {elapsed:.2f}s")


# Minimal settings dict matching what the script expects
settings = {
    "seed": 42,
    "num_clients": 3,
    "partition": {
        "client_0": 0,
        "client_1": 1,
        "client_2": 2,
    },
    "round": 1,
    "train_rounds_interval": {
        "0": (0, 5),
        "1": (0, 5),
        "2": (0, 5),
    },
    "dirichlet_alpha": 0.3,
}

CLIENT_ID = "client_0"
BATCH_SIZE = 16
ALPHA = 0.3


def main():
    print("=" * 50)
    print(f"Testing make_loaders_for_client for {CLIENT_ID}")
    print("=" * 50)

    with timer("make_loaders_for_client (total)"):

        with timer("  -> load_imagenette train + val (inside make_loaders)"):
            # This is implicitly called inside make_loaders_for_client,
            # so we time the whole call and break down sub-steps manually below
            pass

        with timer("  -> full make_loaders_for_client call"):
            train_loader, val_loader, n_samples = make_loaders_for_client(
                settings=settings,
                client_id=CLIENT_ID,
                batch_size=BATCH_SIZE,
                alpha=ALPHA,
            )

    print(f"\nTrain samples for {CLIENT_ID}: {n_samples}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Time iterating one full epoch of train
    print()
    with timer("Iterate full train_loader (1 epoch)"):
        for i, (images, labels) in enumerate(train_loader):
            if i == 0:
                print(f"  First batch — images: {images.shape}, labels: {labels.shape}")

    # Time iterating full val loader
    print()
    with timer("Iterate full val_loader"):
        for i, (images, labels) in enumerate(val_loader):
            if i == 0:
                print(f"  First batch — images: {images.shape}, labels: {labels.shape}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
