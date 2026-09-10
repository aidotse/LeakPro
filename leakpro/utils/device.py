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

_HPU_IMPORT_ERROR: Optional[str] = None
_hthpu = None
_htcore = None
_habana_is_lazy = None

try:  # pragma: no cover - exercised only on Habana systems
    import habana_frameworks.torch.hpu as _hthpu  # type: ignore[import-not-found]
    try:
        import habana_frameworks.torch.core as _htcore  # type: ignore[import-not-found]
    except ImportError as exc:
        _htcore = None
        _HPU_IMPORT_ERROR = f"habana_frameworks.torch.core unavailable: {exc}"
    try:
        from habana_frameworks.torch.utils.internal import (
            is_lazy as _habana_is_lazy,  # type: ignore[import-not-found]  # noqa: E501
        )
    except ImportError:
        _habana_is_lazy = None
except ImportError as exc:
    _hthpu = None
    _htcore = None
    _HPU_IMPORT_ERROR = str(exc)


def _detect_hpu_lazy_mode() -> bool:
    """Determine whether the installed Habana stack is running in lazy mode.

    Habana's own ``is_lazy()`` treats an *unset* ``PT_HPU_LAZY_MODE`` as eager
    mode (``os.getenv("PT_HPU_LAZY_MODE", "0") != "0"``). An earlier version of
    this module re-implemented that check with the opposite default ("1"),
    so whenever the variable was left unset it assumed lazy mode while the
    installed Habana runtime was actually running eager — ``mark_step()``
    would then call ``_htcore.mark_step()`` needlessly, which Habana itself
    silently downgrades to a no-op with a one-time warning. Deferring to
    Habana's own helper (when importable) makes this immune to Habana
    changing its default again in a future release; the env-var fallback
    below mirrors Habana's current default for hosts where the helper isn't
    available.
    """
    if _habana_is_lazy is not None:
        try:
            return bool(_habana_is_lazy())
        except Exception as exc:  # noqa: BLE001
            logger.warning("habana_frameworks.is_lazy() raised: %s; falling back to PT_HPU_LAZY_MODE.", exc)
    return os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"


# PT_HPU_LAZY_MODE=0 (Habana's own default when unset) means eager mode —
# mark_step() is a no-op in that mode.
_HPU_LAZY_MODE: bool = _detect_hpu_lazy_mode()


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


@lru_cache(maxsize=None)
def _detect_device(override: str) -> torch.device:
    """Do the actual detection/override work for a given ``LEAKPRO_DEVICE`` value.

    Cached per distinct ``override`` string (``""`` means "no override, run normal
    detection"). Keying the cache on ``override`` — rather than the parameter-less
    ``@lru_cache`` this used to carry directly on :func:`get_device` — means a
    ``LEAKPRO_DEVICE`` set *after* the first call (e.g. after ``import leakpro``,
    whose banner already calls :func:`get_device` once) still takes effect: it's a
    cache miss on a new key instead of silently returning whatever was cached under
    the old value. Call :func:`get_device.cache_clear` to drop all cached values.
    """
    if override:
        if override in _VALID_OVERRIDE_DEVICES:
            logger.info("Hardware detection: device overridden by LEAKPRO_DEVICE=%s.", override)
            if override == "hpu":
                if _hthpu is None:
                    raise HPUAcquisitionError(
                        "LEAKPRO_DEVICE=hpu was requested but habana_frameworks is not installed.",
                    )
                require_hpu()  # raises HPUAcquisitionError if installed but unusable
            elif override == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "LEAKPRO_DEVICE=cuda was requested but torch.cuda.is_available() is False.",
                )
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


def get_device() -> torch.device:
    """Return the best available ``torch.device`` for this host.

    Reads ``LEAKPRO_DEVICE`` fresh on every call and dispatches to
    :func:`_detect_device`, which caches per distinct override value. This means the
    result is stable within a given ``LEAKPRO_DEVICE`` setting (detection only runs
    once per value) but a *change* to the env var — e.g. code that sets it after
    ``import leakpro`` has already run detection once via its startup banner —
    is picked up on the next call rather than silently ignored.

    The environment variable ``LEAKPRO_DEVICE`` can pin the device without
    code changes — useful in CI or when debugging on a mixed HPU+CUDA machine::

        LEAKPRO_DEVICE=cpu pytest ...

    Accepted values (case-insensitive): ``cpu``, ``cuda``, ``hpu``. An explicit
    override is never silently downgraded: if the requested device isn't actually
    usable, this raises immediately instead of returning a device that would fail
    later inside a ``.to()`` call. An unrecognised value is ignored with a warning
    and normal detection runs.

    Raises:
        HPUAcquisitionError: ``LEAKPRO_DEVICE=hpu`` was requested but
            ``habana_frameworks`` isn't installed, or is installed but no usable
            HPU could be acquired — see that class's docstring. Also raised by
            normal (non-override) detection when the Habana stack is installed
            but unusable.
        RuntimeError: ``LEAKPRO_DEVICE=cuda`` was requested but
            ``torch.cuda.is_available()`` is ``False``.

    """
    override = os.environ.get("LEAKPRO_DEVICE", "").strip().lower()
    return _detect_device(override)


# Forward the full lru_cache introspection API so get_device keeps behaving like a
# directly-decorated lru_cache function (callers/diagnostics may reasonably use any
# of these, not just cache_clear).
get_device.cache_clear = _detect_device.cache_clear  # type: ignore[attr-defined]
get_device.cache_info = _detect_device.cache_info  # type: ignore[attr-defined]
get_device.cache_parameters = _detect_device.cache_parameters  # type: ignore[attr-defined]


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


def hpu_is_installed() -> bool:
    """Return whether the ``habana_frameworks`` package was importable at process start.

    Unlike :func:`require_hpu`, this never raises and says nothing about whether a
    physical HPU can actually be acquired right now — it only checks that the
    Habana *library* is present. Intended for callers that must not fail just
    because a card can't be acquired (e.g. RNG seeding), and so shouldn't go
    through :func:`get_device`'s acquisition probe.
    """
    return _hthpu is not None
