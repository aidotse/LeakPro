#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Init file for leakpro package.

Public symbols (``LeakPro``, ``AbstractInputHandler``) are exposed via PEP 562
lazy ``__getattr__`` so that ``import leakpro`` succeeds even when torch is
missing or in a broken state (e.g. PyTorch/Habana version mismatch). This keeps
``python -m leakpro.utils.hardware_setup`` usable as a diagnostic on a broken
install.
"""
import os as _os
from typing import Any as _Any


def __getattr__(name: str) -> _Any:
    """Lazily import the public API on first access."""
    if name == "AbstractInputHandler":
        from .input_handler.abstract_input_handler import AbstractInputHandler

        return AbstractInputHandler
    if name == "LeakPro":
        from .leakpro import LeakPro

        return LeakPro
    raise AttributeError(f"module 'leakpro' has no attribute {name!r}")


__all__ = ["AbstractInputHandler", "LeakPro"]


# Best-effort platform banner on first import. Wrapped in a broad try/except so
# a broken torch/Habana install never blocks `import leakpro` -- the CLI in
# `leakpro.utils.hardware_setup` is the place to surface those errors. We use
# `device.get_device()` directly (rather than `hardware_setup.detect_profile`)
# so that running `python -m leakpro.utils.hardware_setup` doesn't pre-import
# its own __main__ module via this banner path.
if _os.environ.get("LEAKPRO_QUIET_DEVICE_BANNER") != "1":
    try:
        from .utils.device import get_device as _get_device
        from .utils.logger import logger as _logger

        _device = _get_device()
        _logger.info("LeakPro device profile: %s", _device.type)
        if _device.type == "hpu":
            _logger.info(
                "HPU requires `habana-torch-plugin` and `lightning-habana`; "
                "run `python -m leakpro.utils.hardware_setup --install` for guided setup."
            )
    except Exception as _banner_err:  # noqa: BLE001 - banner must never break imports
        try:
            from .utils.logger import logger as _logger

            _logger.warning(
                "LeakPro device detection skipped (%s). "
                "Run `python -m leakpro.utils.hardware_setup` for diagnostics.",
                _banner_err,
            )
        except Exception:  # noqa: BLE001 - nothing else we can do
            pass
