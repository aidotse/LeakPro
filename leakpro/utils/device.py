#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Hardware detection utility for selecting the runtime device.

Selection order:
    1. Habana Gaudi HPU (if ``habana_frameworks.torch`` is installed and reports an
       available device).
    2. NVIDIA CUDA (if ``torch.cuda.is_available()``).
    3. CPU.

All imports of ``habana_frameworks`` are guarded so this module is safe to import
on CPU-only or NVIDIA-only systems where the Habana SDK is not installed.
"""
import os
from functools import lru_cache
from typing import Optional

import torch

from leakpro.utils.logger import logger

# PT_HPU_LAZY_MODE=0 means eager mode — mark_step() is a no-op in that mode.
_HPU_LAZY_MODE: bool = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"

_HPU_IMPORT_ERROR: Optional[str] = None
_hthpu = None
_htcore = None

try:  # pragma: no cover - exercised only on Habana systems
    import habana_frameworks.torch.hpu as _hthpu  # type: ignore[import-not-found]
    try:
        import habana_frameworks.torch.core as _htcore  # type: ignore[import-not-found]
    except ImportError as exc:
        _htcore = None
        _HPU_IMPORT_ERROR = f"habana_frameworks.torch.core unavailable: {exc}"
except ImportError as exc:
    _hthpu = None
    _htcore = None
    _HPU_IMPORT_ERROR = str(exc)


def _probe_hpu_acquisition() -> None:
    """Attempt a real HPU allocation; raises if the device cannot be acquired.

    ``_hthpu.is_available()`` only confirms the Habana software stack is loaded —
    it does not guarantee a physical device can actually be claimed (e.g. no card
    present, driver not loaded, device held by another process). This exercises
    the same lazy-init path a later ``tensor.to("hpu")`` call would hit, so a
    missing/busy card is caught here instead of failing deep inside training.

    The ``.cpu()`` is required, not decorative: in Habana's lazy mode (the default)
    ``torch.zeros(1, device="hpu")`` alone only queues the op instead of running it,
    so a dead card would pass this check. Copying the result back forces the queued
    graph to actually execute.
    """
    torch.zeros(1, device="hpu").cpu()


class HPUAcquisitionError(RuntimeError):
    """Habana software stack is installed but no usable HPU could be acquired.

    ``habana_frameworks`` is never installed by accident, so if it's present and a
    device still can't be claimed, that's an environment problem (no card, driver
    not loaded, device held by another process) — not a reason to silently fall
    back to CUDA/CPU. Set ``LEAKPRO_DEVICE=cpu`` to run on CPU deliberately.
    """


def require_hpu() -> bool:
    """Return ``True`` when a Habana Gaudi HPU is usable in this process.

    Named ``require_`` rather than ``is_`` because this is not a plain predicate: it
    only returns ``False`` when ``habana_frameworks`` isn't installed at all — that's
    the sole case a fallback to CUDA/CPU should happen without complaint. If the
    package IS installed but a device can't actually be acquired, this raises
    :class:`HPUAcquisitionError` instead of returning ``False``.
    """
    if _hthpu is None:
        return False
    try:
        available = _hthpu.is_available()
    except Exception as exc:
        raise HPUAcquisitionError(
            f"habana_frameworks is installed but is_available() raised: {exc}",
        ) from exc
    if not available:
        raise HPUAcquisitionError(
            "habana_frameworks is installed but reports no available HPU device.",
        )
    try:
        _probe_hpu_acquisition()
    except Exception as exc:
        raise HPUAcquisitionError(
            f"habana_frameworks is installed but the HPU device could not be acquired: {exc}",
        ) from exc
    return True


_VALID_OVERRIDE_DEVICES = {"cpu", "cuda", "hpu"}


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Return the best available ``torch.device`` for this host.

    The result is cached for the lifetime of the process; call
    :func:`get_device.cache_clear` if you need to re-detect (e.g. in tests).

    The environment variable ``LEAKPRO_DEVICE`` can pin the device without
    code changes — useful in CI or when debugging on a mixed HPU+CUDA machine::

        LEAKPRO_DEVICE=cpu pytest ...

    Accepted values (case-insensitive): ``cpu``, ``cuda``, ``hpu``.
    An unrecognised value is ignored with a warning and normal detection runs.

    Raises:
        HPUAcquisitionError: ``habana_frameworks`` is installed but no usable HPU
            could be acquired. This is deliberate — see that class's docstring.
            Set ``LEAKPRO_DEVICE=cpu`` to opt into CPU instead.

    """
    override = os.environ.get("LEAKPRO_DEVICE", "").strip().lower()
    if override:
        if override in _VALID_OVERRIDE_DEVICES:
            logger.info("Hardware detection: device overridden by LEAKPRO_DEVICE=%s.", override)
            return torch.device(override)
        logger.warning(
            "LEAKPRO_DEVICE=%r is not a recognised device (valid: %s); ignoring override.",
            override,
            ", ".join(sorted(_VALID_OVERRIDE_DEVICES)),
        )
    if require_hpu():
        logger.info("Hardware detection: using Habana Gaudi HPU.")
        return torch.device("hpu")
    if torch.cuda.is_available():
        logger.info("Hardware detection: using NVIDIA CUDA.")
        return torch.device("cuda")
    logger.info("Hardware detection: using CPU.")
    return torch.device("cpu")


def mark_step(device: Optional[torch.device] = None) -> None:
    """Trigger a Habana lazy-mode graph compile/execute boundary.

    No-op when not running on HPU, when ``habana_frameworks.torch.core`` is not
    installed, or when running in HPU eager mode (PT_HPU_LAZY_MODE=0).
    Safe to call unconditionally from device-agnostic code paths.
    """
    if _htcore is None or not _HPU_LAZY_MODE:
        return
    target = device if device is not None else get_device()
    if getattr(target, "type", None) != "hpu":
        return
    _htcore.mark_step()


def hpu_import_error() -> Optional[str]:
    """Return the captured import error string for diagnostics, if any."""
    return _HPU_IMPORT_ERROR
