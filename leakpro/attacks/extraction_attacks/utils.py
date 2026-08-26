#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Input validation, hashing, and image-range utilities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor


@contextmanager
def seeded_torch_rng(device: torch.device, seed: int) -> Iterator[None]:
    """Seed CPU and one selected accelerator, then restore their RNG states."""
    cpu_state = torch.random.get_rng_state()
    accelerator_state: Tensor | None = None
    accelerator_index: int | None = None
    if device.type == "cuda":
        accelerator_index = device.index if device.index is not None else torch.cuda.current_device()
        accelerator_state = torch.cuda.get_rng_state(accelerator_index)
    elif device.type == "mps":
        accelerator_state = torch.mps.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            if accelerator_index is None:
                raise RuntimeError("CUDA RNG device was not resolved.")
            with torch.cuda.device(accelerator_index):
                torch.cuda.manual_seed(seed)
        elif device.type == "mps":
            torch.mps.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if device.type == "cuda" and accelerator_state is not None and accelerator_index is not None:
            torch.cuda.set_rng_state(accelerator_state, accelerator_index)
        elif device.type == "mps" and accelerator_state is not None:
            torch.mps.set_rng_state(accelerator_state)


def require_authorized(authorized_audit: bool) -> None:
    """Fail closed unless the caller confirms audit authorization."""
    if not authorized_audit:
        raise PermissionError(
            "authorized_audit must be true. Run extraction only on models and data you are authorized to audit."
        )


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic torch device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid device string: {requested!r}.") from error
    return _validate_explicit_device(device)


def _validate_explicit_device(device: torch.device) -> torch.device:
    """Reject unsupported or unavailable explicit extraction devices."""
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported extraction device type: {device.type!r}.")
    if device.type == "cpu" and device.index is not None:
        raise ValueError("CPU extraction devices must not include an index.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device.index} is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    if device.type == "mps" and device.index not in {None, 0}:
        raise ValueError(f"MPS device index {device.index} is unavailable.")
    return device


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


def stable_hash(value: BaseModel | dict[str, Any], length: int = 12) -> str:
    """Hash a validated config using canonical JSON."""
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def condition_fingerprint(condition: object) -> str:
    """Return a non-reversible identifier for a condition without storing it."""
    payload = json.dumps(
        _canonical_condition(condition),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_condition(value: object) -> object:
    """Encode supported condition values without process-dependent representations."""
    if isinstance(value, BaseModel):
        model_name = f"{type(value).__module__}.{type(value).__qualname__}"
        return {
            "type": "model",
            "model": model_name,
            "value": _canonical_condition(value.model_dump(mode="json")),
        }
    if isinstance(value, Tensor):
        return _canonical_tensor_condition(value)
    if isinstance(value, np.ndarray):
        return _canonical_array_condition(value)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Extraction condition dictionaries require string keys.")
        return {"type": "dict", "value": {key: _canonical_condition(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        container_type = "list" if isinstance(value, list) else "tuple"
        return {"type": container_type, "value": [_canonical_condition(item) for item in value]}
    return _canonical_scalar_condition(value)


def _canonical_tensor_condition(value: Tensor) -> dict[str, object]:
    """Encode a tensor using stable dtype, shape, and byte content."""
    tensor = value.detach().to(device="cpu").contiguous()
    if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
        raise ValueError("Extraction conditions must not contain NaN or infinity.")
    payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes().hex()
    return {"type": "tensor", "dtype": str(tensor.dtype), "shape": list(tensor.shape), "bytes": payload}


def _canonical_array_condition(value: np.ndarray) -> dict[str, object]:
    """Encode a non-object NumPy array using stable metadata and bytes."""
    if value.dtype.hasobject:
        raise TypeError("Object arrays are not supported as extraction conditions.")
    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise ValueError("Extraction conditions must not contain NaN or infinity.")
    contiguous = np.ascontiguousarray(value)
    return {
        "type": "ndarray",
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "bytes": contiguous.tobytes().hex(),
    }


def _canonical_scalar_condition(value: object) -> dict[str, object]:
    """Encode supported scalar condition types or reject the value."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Extraction conditions must not contain NaN or infinity.")
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(
        "Unsupported extraction condition type. Use strings, finite scalars, bytes, lists, tuples, string-keyed "
        "dictionaries, Pydantic models, NumPy arrays, or torch tensors."
    )


def extraction_audit_fingerprint(
    target_fingerprint: str,
    *,
    conditions: Sequence[Any] | None,
    reference_images: Tensor | None,
) -> str:
    """Hash target identity and attack inputs without persisting their contents."""
    if not target_fingerprint.strip():
        raise ValueError("target fingerprint must not be empty.")
    digest = hashlib.sha256()

    def update(tag: str, payload: bytes) -> None:
        digest.update(tag.encode("ascii"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    update("target", target_fingerprint.encode("utf-8"))
    if conditions is None:
        update("conditions", b"none")
    else:
        for index, condition in enumerate(conditions):
            update(f"condition:{index}", condition_fingerprint(condition).encode("ascii"))
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
