"""Image augmentation extension for dataloader outputs already normalized."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F  # noqa: N812

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler


def _build_registry_mild() -> Dict[str, transforms.Transform]:
    bil = InterpolationMode.BILINEAR
    R = {
        # photometric (very mild)
        "cj_tiny": transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        "blur_s":  transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
        # geometric
        "hflip":   transforms.RandomHorizontalFlip(p=1.0),
        "rot3":    transforms.RandomRotation(degrees=3, interpolation=bil),
        "aff_tiny": transforms.RandomAffine(
            degrees=0, translate=(2/32, 2/32), scale=(0.97, 1.03),
            shear=(-3, 3, -3, 3), interpolation=bil),
        "zoom_cc": transforms.Compose([
            transforms.Resize((34, 34), interpolation=bil),
            transforms.CenterCrop(32),
        ]),
        "rcrop_p2": transforms.RandomCrop(32, padding=2),
        # NEW: CIFAR-style reflect crop
        "rcrop_p4_reflect": transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    }
    # curated combos
    R["flip_cj"]  = transforms.Compose([R["hflip"],   R["cj_tiny"]])
    R["rot_cj"]   = transforms.Compose([R["rot3"],    R["cj_tiny"]])
    R["aff_blur"] = transforms.Compose([R["aff_tiny"], R["blur_s"]])
    R["zoom_cj"]  = transforms.Compose([R["zoom_cc"], R["cj_tiny"]])
    R["crop_cj"]  = transforms.Compose([R["rcrop_p2"], R["cj_tiny"]])
    return R


def _policy_tiers() -> Dict[str, List[str]]:
    easy = [
        "rcrop_p4_reflect",  # CIFAR-style pad+crop
        "hflip",
    ]
    medium = easy + [
        "cj_tiny",
        "rot3",
    ]
    # Strong adds a couple more shape/photometric ops; keep size-preserving overall
    strong = medium + [
        "aff_tiny",
        "blur_s",
        # Optionally one of the curated combos to emulate a RandAugment-ish mix:
        "flip_cj",
    ]
    return {"easy": easy, "medium": medium, "strong": strong}

def _to_pil_from_normalized(x: Tensor,
                            mean: Tensor,
                            std: Tensor,
                            force_rgb: bool = True) -> Image.Image:
    """Convert a normalized CHW tensor -> raw [0,1] tensor -> PIL (RGB/L)."""
    assert x.ndim == 3, f"Expected CHW single image, got {tuple(x.shape)}"
    C, H, W = x.shape  # noqa: N806
    dev = x.device
    m = mean.detach().to(dtype=x.dtype, device=dev)[:C].view(C, 1, 1)
    s = std.detach().to(dtype=x.dtype, device=dev)[:C].view(C, 1, 1)
    un = (x * s + m).clamp(0, 1).cpu()   # <-- clamp to [0,1]
    if C == 1 and force_rgb:
        un = un.expand(3, H, W)
    return F.to_pil_image(un)

def _from_pil_to_raw01(pil: Image.Image,
                       target_channels: int,
                       device: torch.device) -> Tensor:
    """Convert PIL -> raw [0,1] CHW tensor (no normalization)."""
    pil = pil.convert("L") if target_channels == 1 else pil.convert("RGB")
    t = F.to_tensor(pil)  # [C,H,W] in [0,1], float32 CPU
    if target_channels == 1 and t.shape[0] == 3:
        t = t.mean(0, keepdim=True)
    return t.to(device)

class ImageAugmentor:
    """Augment already-normalized inputs; return raw [0,1] tensors (later normalized by dataset)."""

    def __init__(self, handler: AbstractInputHandler, augment_strength: str = "easy", force_rgb: bool = True) -> None:
        self.mean = handler.population.mean
        self.std = handler.population.std
        self.force_rgb = force_rgb
        self._reg = _build_registry_mild()
        self._tiers = _policy_tiers()
        self.set_augment_strength(augment_strength)

    def set_augment_strength(self, augment_strength: str) -> None:
        """Set the augmentation strength level.

        Args:
            augment_strength: One of "none", "easy", "medium", "strong".

        """

        if augment_strength not in {"none", "easy", "medium", "strong"}:
            raise ValueError(f"Unknown strength: {augment_strength}")
        self._strength = augment_strength

    def _augment(self, num_ops:int) -> callable:
        """Returns a callable that:
        - takes a normalized CHW tensor
        - unnormalizes -> PIL
        - applies tiered raw-space augs
        - returns raw [0,1] CHW tensor (no normalization).
        """  # noqa: D205
        if self._strength == "none":
            def fn(x: Tensor, _seed: int):
                pil = _to_pil_from_normalized(x, self.mean, self.std, force_rgb=self.force_rgb)
                y = _from_pil_to_raw01(pil, x.shape[0], x.device).to(x.dtype)
                return y, []
            return fn

        keys_all = self._tiers[self._strength]
        pool = [self._reg[k] for k in keys_all]
        L = len(pool)

        def fn(x: Tensor, seed: int):
            assert x.ndim == 3, "Expected Tensor[C,H,W]"
            C, H, W = x.shape

            # deterministic start index from seed (no RNG state pollution)
            start = (seed * 2654435761) % L
            # choose num_ops ops cyclically from the tier
            sel_idxs = [(start + i) % L for i in range(max(1, num_ops))]
            sel_names = [keys_all[i] for i in sel_idxs]
            chain = transforms.Compose([pool[i] for i in sel_idxs])

            pil = _to_pil_from_normalized(x, self.mean, self.std, force_rgb=self.force_rgb)
            pil = chain(pil)
            y = _from_pil_to_raw01(pil, target_channels=C, device=x.device).to(x.dtype)
            if y.shape[-2:] != (H, W):
                y = transforms.functional.resize(y, [H, W], antialias=True)
            return y, sel_names

        return fn

    def augment(
        self,
        x: Tensor,
        k: int,
        *,
        num_ops: int = 2,
        base_seed: int = 0,
        stack: bool = True,
    ) -> Tuple[Any, Optional[List[List[str]]]]:
        """Apply k augmentations to a single normalized CHW tensor.

        Args:
            x: Normalized CHW tensor.
            k: Number of augmentations to produce (k=0 means return raw [0,1] version).
            base_seed: Base seed for RNG (not used in current deterministic ops, but kept for API symmetry).
            stack: If True, stack outputs into a single tensor of shape (k,C,H,W). If False, return a list of k tensors.

        Returns:
            - If stack=True: Tensor[k,C,H,W] of augmented raw [0,1] images.
            - If stack=False: List of k tensors, each CHW raw [0,1].
            - If k=0: Tensor[1,C,H,W] or Tensor[C,H,W] (if stack=False) of raw [0,1] image (no aug).
            - If k>0: Also returns List of List of str, the ops used for each augmentation.

        """
        if k == 0:
            # Return the original image converted to raw [0,1] (dataset will normalize later)
            pil = _to_pil_from_normalized(x, self.mean, self.std, force_rgb=self.force_rgb)
            x_return = _from_pil_to_raw01(pil, x.shape[0], x.device).to(x.dtype)
            x_return = x_return.unsqueeze(0) if stack else x_return

            return x_return, None

        tf = self._augment(num_ops=num_ops)
        ys, ops_used = [], []
        for j in range(k):
            _seed = base_seed + j
            y, ops = tf(x, _seed)
            ys.append(y)
            ops_used.append(ops)
        out = torch.stack(ys, dim=0) if stack else ys
        return out, ops_used
