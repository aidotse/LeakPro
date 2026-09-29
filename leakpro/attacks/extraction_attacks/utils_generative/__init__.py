#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Input validation, hashing, and image-range utilities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor
from tqdm.auto import tqdm


# seed_everything changes process-wide RNG state; sampling must restore the
# previous state so one attack does not change another attack's samples.
@contextmanager
def seeded_torch_rng(device: torch.device, seed: int) -> Iterator[None]:
    """Seed CPU and selected CUDA, then restore their RNG states."""
    cpu_state = torch.random.get_rng_state()
    accelerator_state: Tensor | None = None
    accelerator_index: int | None = None
    if device.type == "cuda":
        accelerator_index = device.index if device.index is not None else torch.cuda.current_device()
        accelerator_state = torch.cuda.get_rng_state(accelerator_index)
    try:
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            if accelerator_index is None:
                raise RuntimeError("CUDA RNG device was not resolved.")
            with torch.cuda.device(accelerator_index):
                torch.cuda.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if device.type == "cuda" and accelerator_state is not None and accelerator_index is not None:
            torch.cuda.set_rng_state(accelerator_state, accelerator_index)


def require_authorized(authorized_audit: bool) -> None:
    """Fail closed unless the caller confirms audit authorization."""
    if not authorized_audit:
        raise PermissionError(
            "authorized_audit must be true. Run extraction only on models and data you are authorized to audit."
        )


def validate_image_batch(
    images: Tensor,
    expected_shape: tuple[int, int, int] | None = None,
    expected_count: int | None = None,
) -> Tensor:
    """Validate a finite BCHW floating-point image batch."""
    if not isinstance(images, Tensor):
        raise TypeError("The diffusion adapter must return a torch.Tensor.")
    if images.ndim != 4:
        raise ValueError(f"Expected a BCHW tensor, got shape {tuple(images.shape)}.")
    if images.shape[0] < 1:
        raise ValueError("Generated image batches must not be empty.")
    if expected_count is not None and images.shape[0] != expected_count:
        raise ValueError(f"Requested {expected_count} images, but the adapter returned {images.shape[0]}.")
    if expected_shape is not None and tuple(images.shape[1:]) != expected_shape:
        raise ValueError(f"Expected image shape {expected_shape}, got {tuple(images.shape[1:])}.")
    if not images.is_floating_point():
        raise TypeError("Generated images must use a floating-point dtype.")
    if not torch.isfinite(images).all():
        raise ValueError("Generated images contain NaN or infinity.")
    return images


def to_zero_one(images: Tensor, image_range: Literal["zero_one", "minus_one_one"]) -> Tensor:
    """Convert a validated image batch to [0, 1] without silently clipping invalid data."""
    tolerance = 1e-4
    low, high = (0.0, 1.0) if image_range == "zero_one" else (-1.0, 1.0)
    observed_low = float(images.detach().amin().cpu())
    observed_high = float(images.detach().amax().cpu())
    if observed_low < low - tolerance or observed_high > high + tolerance:
        raise ValueError(
            f"Image values [{observed_low:.6g}, {observed_high:.6g}] violate configured range [{low}, {high}]."
        )
    normalized = images if image_range == "zero_one" else (images + 1.0) / 2.0
    return normalized.clamp(0.0, 1.0)


def from_zero_one(images: Tensor, image_range: Literal["zero_one", "minus_one_one"]) -> Tensor:
    """Convert [0, 1] images to the target model's configured range."""
    return images if image_range == "zero_one" else images.mul(2.0).sub(1.0)


def encode_uint8(images_zero_one: Tensor) -> Tensor:
    """Encode normalized images compactly and deterministically on CPU."""
    return images_zero_one.mul(255.0).round().to(dtype=torch.uint8, device="cpu")


def decode_uint8(images: Tensor, image_range: Literal["zero_one", "minus_one_one"]) -> Tensor:
    """Decode compact images into the target model's range."""
    zero_one = images.to(dtype=torch.float32).div(255.0)
    return from_zero_one(zero_one, image_range)


def batch_ranges(total: int, batch_size: int) -> Iterator[tuple[int, int]]:
    """Yield half-open deterministic batch ranges."""
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def progress_batches(total: int, batch_size: int, description: str) -> Iterator[tuple[int, int]]:
    """Count completed samples, including a final partial batch."""
    with tqdm(total=total, desc=description, unit="sample", dynamic_ncols=True) as progress:
        for start, end in batch_ranges(total, batch_size):
            yield start, end
            progress.update(end - start)


def normalize_conditions(conditions: Sequence[Any] | None) -> list[Any] | None:
    """Copy an ordered condition collection without splitting scalar values."""
    if conditions is None:
        return None
    if isinstance(conditions, (str, bytes, bytearray, memoryview, Mapping)) or not isinstance(conditions, Sequence):
        raise TypeError("Extraction conditions must be an ordered sequence of prompts or labels.")
    return list(conditions)


def extraction_audit_hash(
    target_hash: str,
    *,
    reference_images: Tensor | None,
) -> str:
    """Hash target identity and attack inputs without persisting their contents."""
    if not target_hash.strip():
        raise ValueError("target hash must not be empty.")
    digest = hashlib.sha256()

    def update(tag: str, payload: bytes) -> None:
        digest.update(tag.encode("ascii"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    update("target", target_hash.encode("utf-8"))
    if reference_images is None:
        update("references", b"none")
    else:
        metadata = json.dumps(
            {"dtype": str(reference_images.dtype), "shape": list(reference_images.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        update("reference_metadata", metadata)
        for start, end in batch_ranges(reference_images.shape[0], 16):
            block = reference_images[start:end].detach().to(device="cpu").contiguous().view(torch.uint8)
            update(f"reference_block:{start}", block.numpy().tobytes())
    return digest.hexdigest()


def json_safe(value: object) -> object:
    """Convert common scientific Python values into strict JSON-compatible values."""
    if isinstance(value, BaseModel):
        converted: object = json_safe(value.model_dump(mode="json"))
    elif isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Result metadata dictionaries require string keys.")
        converted = {key: json_safe(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        converted = [json_safe(item) for item in value]
    elif isinstance(value, np.ndarray):
        converted = json_safe(value.tolist())
    elif isinstance(value, np.generic):
        converted = json_safe(value.item())
    elif isinstance(value, Tensor):
        converted = json_safe(value.detach().cpu().tolist())
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Result metadata must not contain NaN or infinity.")
    elif value is None or isinstance(value, (str, int, float, bool)):
        converted = value
    else:
        raise TypeError(f"Unsupported result metadata type: {type(value).__name__}.")
    return converted
