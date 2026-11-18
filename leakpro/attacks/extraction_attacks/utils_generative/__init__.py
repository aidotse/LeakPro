#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Input validation, hashing, and image-range utilities."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor, nn
from tqdm.auto import tqdm

from leakpro.utils.save_load import hash_model


# seed_everything changes process-wide RNG state; sampling must restore the
# previous state so one attack does not change another attack's samples.
@contextmanager
def seeded_torch_rng(device: torch.device, seed: int) -> Iterator[None]:
    """Seed CPU and the selected accelerator, then restore their RNG states."""
    cpu_state = torch.random.get_rng_state()
    accelerator_state: Tensor | None = None
    accelerator_index: int | None = None
    hpu_random = import_module("habana_frameworks.torch.hpu.random") if device.type == "hpu" else None
    hpu_states = hpu_random.get_rng_state_all() if hpu_random is not None else None
    mps_state = torch.mps.get_rng_state(device) if device.type == "mps" else None
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
        if hpu_random is not None:
            hpu_random.manual_seed_all(seed)
        if mps_state is not None:
            torch.mps.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if device.type == "cuda" and accelerator_state is not None and accelerator_index is not None:
            torch.cuda.set_rng_state(accelerator_state, accelerator_index)
        if hpu_random is not None and hpu_states is not None:
            hpu_random.set_rng_state_all(hpu_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state, device)


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
    normalized = list(conditions)
    if any(isinstance(condition, bool) or not isinstance(condition, (str, int, np.integer)) for condition in normalized):
        raise TypeError("Extraction conditions support only string prompts and integer labels; tensors/arrays are unsupported.")
    return normalized


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


def module_metadata(
    module: nn.Module, config: dict[str, Any] | None, *, include_state_hash: bool = True,
) -> dict[str, Any]:
    """Combine declared behavior with module state; weights alone omit non-state settings.

    The caller must describe architecture, nested settings and behavior revisions
    accurately. Arbitrary Python behavior cannot be inferred from a module's type.
    """
    if not isinstance(config, dict) or not config:
        raise ValueError("Module identity requires a non-empty configuration describing its effective behavior.")
    metadata = {
        "type": f"{type(module).__module__}.{type(module).__qualname__}",
        "config": json_safe(config),
        "state_layout": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in module.state_dict().items()
        },
    }
    if include_state_hash:
        metadata["state_hash"] = hash_model(module)
    return metadata


def read_extraction_metadata(
    getter: Callable[..., dict[str, Any] | None], *, conditional: bool, guided: bool | None = None,
) -> dict[str, Any] | None:
    """Pass conditioning context when the metadata getter accepts it."""
    kwargs = {} if guided is None else {"guided": guided}
    try:
        signature = inspect.signature(getter)
        if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            signature.bind_partial(conditional=conditional)
    except (TypeError, ValueError):
        pass
    else:
        kwargs["conditional"] = conditional
    return getter(**kwargs)


def adapter_metadata(
    adapter: object, declared: dict[str, Any] | None, *, guided: bool, conditional: bool = True,
) -> dict[str, Any]:
    """Resolve provider/adapter settings before generation and bind the actual model state."""
    if declared is None:
        getter = getattr(adapter, "get_extraction_metadata", None)
        if not callable(getter):
            raise ValueError("Provide generation_metadata or adapter.get_extraction_metadata() for result identity.")
        declared = read_extraction_metadata(getter, guided=guided, conditional=conditional)
    if not isinstance(declared, dict) or not isinstance(declared.get("sampler"), dict) or not declared["sampler"]:
        raise ValueError("Extraction metadata requires non-empty target and sampler configurations.")
    return {
        # hash_attack already hashes the target's weights; do not read them twice.
        "target": module_metadata(adapter.model, declared.get("target"), include_state_hash=False),
        "sampler": json_safe(declared["sampler"]),
        "image_shape": list(adapter.image_shape),
    }
