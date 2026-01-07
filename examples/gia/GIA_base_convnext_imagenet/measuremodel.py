"""Inverting on a single image (ConvNeXt)."""
import optuna
from torchvision.models import convnext_tiny, convnext_base, swin_t, Swin_T_Weights, vit_b_16, swin_v2_t, maxvit_t
from model import convnext_tiny_cifar10
from imagenet import get_imagenette_loader, get_cifar10_loader

from leakpro.attacks.gia_attacks.gia_estimate import GIABase, GIABaseConfig
from leakpro.fl_utils.data_utils import GiaImageCloneNoiseExtension
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

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=float, required=True, help="pixel_noise_p value, e.g. 0.7")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--start-idx", type=int, default=19)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--at-iterations", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    exp = float(args.exp)

    seed_everything(args.seed)

    # If needed for ViT-B, leave commented or gate behind a flag.
    # if torch.cuda.is_available():
    #     torch.backends.cuda.enable_flash_sdp(False)
    #     torch.backends.cuda.enable_mem_efficient_sdp(False)
    #     torch.backends.cuda.enable_math_sdp(True)

    model = convnext_tiny_cifar10(small_stem=True)
    model.eval()

    NUM_IMAGES = 1
    BATCH_SIZE = 1
    NUM_CLIENT_LOADERS = 1
    PROXIES_PER_CLIENT = 1
    NUM_WORKERS = args.num_workers
    START_IDX = args.start_idx

    trial_data, client_dataloader, proxy_loader, data_mean, data_std = build_trial_data(
        num_client_loaders=NUM_CLIENT_LOADERS,
        proxies_per_client=PROXIES_PER_CLIENT,
        num_images=NUM_IMAGES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        start_idx=START_IDX,
    )

    config = GIABaseConfig(
        data_extension=GiaImageCloneNoiseExtension(pixel_noise_p=exp),
        chose_best_ssim_as_final=False,
        at_iterations=args.at_iterations,
    )

    attack_object = GIABase(
        model,
        client_dataloader,
        data_mean,
        data_std,
        proxy_loader=proxy_loader,
        optuna_trial_data=trial_data,
        configs=config,
        exp_name=f"exp_{exp}",
    )

    optuna_config = OptunaConfig(n_trials=args.n_trials)
    attack_object.run_with_optuna(optuna_config)


# if __name__ == "__main__":
#     exps = [i / 100 for i in range(70, 101, 5)]
#     for exp in exps:
#         seed_everything(1234)
#         # if torch.cuda.is_available(): # needed for vit-b
#         #     torch.backends.cuda.enable_flash_sdp(False)
#         #     torch.backends.cuda.enable_mem_efficient_sdp(False)
#         #     torch.backends.cuda.enable_math_sdp(True)
#         model = convnext_tiny_cifar10(small_stem=True)
#         #model = convnext_tiny(weights=None, num_classes=1000)
#         model.eval()

#         NUM_IMAGES = 1
#         BATCH_SIZE = 1
#         NUM_CLIENT_LOADERS = 1
#         PROXIES_PER_CLIENT = 1
#         NUM_WORKERS = 2
#         START_IDX = 19

#         trial_data, client_dataloader, proxy_loader, data_mean, data_std = build_trial_data(
#             num_client_loaders=NUM_CLIENT_LOADERS,
#             proxies_per_client=PROXIES_PER_CLIENT,
#             num_images=NUM_IMAGES,
#             batch_size=BATCH_SIZE,
#             num_workers=NUM_WORKERS,
#             start_idx=START_IDX,
#         )
#         config = GIABaseConfig(data_extension=GiaImageCloneNoiseExtension(pixel_noise_p=exp), chose_best_ssim_as_final=False, at_iterations=10000)

#         attack_object = GIABase(
#             model,
#             client_dataloader,
#             data_mean,
#             data_std,
#             proxy_loader=proxy_loader,
#             optuna_trial_data=trial_data,
#             configs=config,
#             exp_name=f"exp_{exp}",
#         )

#         optuna_config = OptunaConfig(n_trials=10)
#         attack_object.run_with_optuna(optuna_config)
