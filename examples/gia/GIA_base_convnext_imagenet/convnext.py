"""Inverting on a single image (ConvNeXt)."""
from functools import partial
import optuna
from torchvision.models import convnext_tiny, convnext_base, swin_t, Swin_T_Weights, vit_b_16, swin_v2_t, maxvit_t
from imagenet import get_imagenette_loader, get_cifar10_loader

from leakpro.attacks.gia_attacks.gia_estimate import GIABase, GIABaseConfig
from leakpro.fl_utils.data_utils import GiaImageCloneNoiseExtension
from leakpro.schemas import OptunaConfig
from leakpro.utils.seed import seed_everything
import torch

import torch
import torch.nn as nn

import torch
from torchvision.models.swin_transformer import ShiftedWindowAttention, ShiftedWindowAttentionV2

def _replace_attn_with_global(attn: torch.nn.Module, dim: int, num_heads: int, global_hw: tuple[int, int]):
    """Replace a Swin attention module with a new one that uses one global window."""
    H, W = global_hw
    window_size = [H, W]
    shift_size = [0, 0]

    # Preserve dropouts
    attn_drop = getattr(attn, "attention_dropout", 0.0)
    drop = getattr(attn, "dropout", 0.0)

    if isinstance(attn, ShiftedWindowAttentionV2):
        new_attn = ShiftedWindowAttentionV2(
            dim=dim,
            window_size=window_size,
            shift_size=shift_size,
            num_heads=num_heads,
            qkv_bias=(attn.qkv.bias is not None),
            proj_bias=(attn.proj.bias is not None),
            attention_dropout=attn_drop,
            dropout=drop,
        )
        # Copy QKV/proj weights (these are window-size independent)
        new_attn.qkv.weight.data.copy_(attn.qkv.weight.data)
        if attn.qkv.bias is not None:
            new_attn.qkv.bias.data.copy_(attn.qkv.bias.data)
        new_attn.proj.weight.data.copy_(attn.proj.weight.data)
        if attn.proj.bias is not None:
            new_attn.proj.bias.data.copy_(attn.proj.bias.data)

        # Copy V2-specific params
        new_attn.logit_scale.data.copy_(attn.logit_scale.data)
        new_attn.cpb_mlp.load_state_dict(attn.cpb_mlp.state_dict())

        return new_attn

    if isinstance(attn, ShiftedWindowAttention):
        new_attn = ShiftedWindowAttention(
            dim=dim,
            window_size=window_size,
            shift_size=shift_size,
            num_heads=num_heads,
            qkv_bias=(attn.qkv.bias is not None),
            proj_bias=(attn.proj.bias is not None),
            attention_dropout=attn_drop,
            dropout=drop,
        )
        # Copy QKV/proj weights
        new_attn.qkv.weight.data.copy_(attn.qkv.weight.data)
        if attn.qkv.bias is not None:
            new_attn.qkv.bias.data.copy_(attn.qkv.bias.data)
        new_attn.proj.weight.data.copy_(attn.proj.weight.data)
        if attn.proj.bias is not None:
            new_attn.proj.bias.data.copy_(attn.proj.bias.data)

        # NOTE: relative_position_bias_table cannot be copied (shape changes). It stays freshly initialized.
        return new_attn

    raise TypeError(f"Unexpected attention type: {type(attn)}")


def make_swin_late_stages_global(
    model: torch.nn.Module,
    image_size: int = 224,
    patch_size: int = 4,
    global_stages: tuple[int, ...] = (2, 3),
):
    """
    Make selected Swin stages use one global attention window (per stage),
    i.e. window_size == (H_stage, W_stage) and shift_size == (0,0).

    Works with torchvision Swin v1 and v2 models.
    """
    # Stage resolutions for square images
    # stage0: image_size/patch_size, then /2 each patch-merge
    base = image_size // patch_size
    stage_hw = [(base // (2**s), base // (2**s)) for s in range(4)]  # [(56,56),(28,28),(14,14),(7,7)] for 224

    # In torchvision, model.features is:
    # [patch_embed, stage0, merge0, stage1, merge1, stage2, merge2, stage3]
    stage_module_indices = {0: 1, 1: 3, 2: 5, 3: 7}

    for s in global_stages:
        idx = stage_module_indices[s]
        stage_seq = model.features[idx]  # nn.Sequential of blocks
        Hs, Ws = stage_hw[s]

        for blk in stage_seq:
            # blk is SwinTransformerBlock or SwinTransformerBlockV2
            dim = blk.norm1.normalized_shape[0]  # channel dim
            num_heads = blk.attn.num_heads
            blk.attn = _replace_attn_with_global(blk.attn, dim=dim, num_heads=num_heads, global_hw=(Hs, Ws))

    return model


class StaticDominatedLayerNorm(nn.Module):
    """
    LayerNorm where the normalization statistics are dominated by a static vector s.
    This makes LN(x) behave ~ affine(x) with almost constant (input-independent) stats.
    """
    def __init__(self, dim: int, eps: float = 1e-6, alpha: float = 10.0, learnable_static: bool = False):
        super().__init__()
        self.eps = eps
        self.alpha = alpha

        # Standard LN affine params
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

        # Static vector that dominates mean/var when alpha is large
        static = torch.randn(dim)  # zero-mean-ish, nontrivial variance
        if learnable_static:
            self.static = nn.Parameter(static)
        else:
            self.register_buffer("static", static)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim) in torchvision ViT blocks
        s = self.alpha * self.static
        y = x + s

        mean = y.mean(dim=-1, keepdim=True)
        var = y.var(dim=-1, unbiased=False, keepdim=True)
        y = (y - mean) / torch.sqrt(var + self.eps)

        return y * self.weight + self.bias

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

        client_loader, data_mean, data_std = get_imagenette_loader(
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
            proxy, _, _ = get_imagenette_loader(
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
    # if torch.cuda.is_available(): # needed for vit-b
    #     torch.backends.cuda.enable_flash_sdp(False)
    #     torch.backends.cuda.enable_mem_efficient_sdp(False)
    #     torch.backends.cuda.enable_math_sdp(True)

    model = maxvit_t(weights=None, num_classes=1000)
    model.eval()

    NUM_IMAGES = 1
    BATCH_SIZE = 1
    NUM_CLIENT_LOADERS = 15
    PROXIES_PER_CLIENT = 1
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

    optuna_config = OptunaConfig(n_trials=100, pruner=optuna.pruners.NopPruner())
    attack_object.run_with_optuna(optuna_config)
