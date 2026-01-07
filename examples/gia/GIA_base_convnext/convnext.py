"""Inverting on a single image (ConvNeXt)."""
from torchvision.models import convnext_tiny

from cifar import get_cifar10_loader

from leakpro.attacks.gia_attacks.gia_estimate import GIABase
from leakpro.schemas import OptunaConfig
from leakpro.utils.seed import seed_everything


def build_trial_data(
    *,
    num_client_loaders: int,
    proxies_per_client: int,
    num_images: int,
    batch_size: int,
    num_workers: int = 2,
    start_idx: int = 0,
):
    """
    Returns:
      trial_data: list[tuple[client_loader, proxy_loader]] with length
                 num_client_loaders * proxies_per_client

    Layout (non-overlapping contiguous blocks of size num_images):
      For each client c:
        client block:  [base, base+num_images)
        proxy blocks:  [base+1*num_images, base+2*num_images), ... up to proxies_per_client
      Next client starts at:
        base + (1 + proxies_per_client) * num_images
    """
    trial_data = []
    step = num_images * (1 + proxies_per_client)

    data_mean = data_std = None
    client_dataloader = None
    proxy_loader = None

    for c in range(num_client_loaders):
        base = start_idx + c * step

        client_loader, data_mean, data_std = get_cifar10_loader(
            start_idx=base,
            num_images=num_images,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # store a default reference for the GIABase ctor (some codepaths expect non-None)
        if client_dataloader is None:
            client_dataloader = client_loader

        for p in range(proxies_per_client):
            proxy_start = base + (p + 1) * num_images
            proxy, _, _ = get_cifar10_loader(
                start_idx=proxy_start,
                num_images=num_images,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            trial_data.append((client_loader, proxy))

            if proxy_loader is None:
                proxy_loader = proxy

    return trial_data, client_dataloader, proxy_loader, data_mean, data_std


if __name__ == "__main__":
    seed_everything(1234)

    model = convnext_tiny(weights=None, num_classes=10)

    # Parameterize these
    NUM_IMAGES = 16*6
    BATCH_SIZE = 16*6
    NUM_CLIENT_LOADERS = 2
    PROXIES_PER_CLIENT = 5
    NUM_WORKERS = 2
    START_IDX = 0

    trial_data, client_dataloader, proxy_loader, data_mean, data_std = build_trial_data(
        num_client_loaders=NUM_CLIENT_LOADERS,
        proxies_per_client=PROXIES_PER_CLIENT,
        num_images=NUM_IMAGES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        start_idx=START_IDX,
    )

    attack_object = GIABase(
        model,
        client_dataloader,
        data_mean,
        data_std,
        proxy_loader=proxy_loader,
        optuna_trial_data=trial_data,
    )

    optuna_config = OptunaConfig(n_trials=100)
    attack_object.run_with_optuna(optuna_config)
