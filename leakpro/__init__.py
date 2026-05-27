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
from __future__ import annotations

import contextlib
import os as _os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Visible to type checkers (so `from leakpro import LeakPro` resolves) but
    # not executed at runtime, keeping the lazy-import contract intact.
    from .input_handler.abstract_input_handler import AbstractInputHandler  # noqa: F401
    from .leakpro import LeakPro  # noqa: F401


def __getattr__(name: str) -> object:
    """Lazily import the public API on first access."""
    if name == "AbstractInputHandler":
        from .input_handler.abstract_input_handler import AbstractInputHandler  # noqa: PLC0415

        return AbstractInputHandler
    if name == "LeakPro":
        from .leakpro import LeakPro  # noqa: PLC0415

        return LeakPro
    raise AttributeError(f"module 'leakpro' has no attribute {name!r}")


__all__ = ["AbstractInputHandler", "LeakPro"]


# Best-effort platform banner on first import. Wrapped in a broad try/except so
# a broken torch/Habana install never blocks ``import leakpro`` -- the CLI in
# ``leakpro.utils.hardware_setup`` is the place to surface those errors. We use
# ``device.get_device()`` directly (rather than ``hardware_setup.detect_profile``)
# so that running ``python -m leakpro.utils.hardware_setup`` doesn't pre-import
# its own __main__ module via this banner path.
def _emit_device_banner() -> None:
    """Log the detected device. Deferred imports keep ``import leakpro`` resilient to a broken torch."""
    try:
        from .utils.device import get_device as _get_device  # noqa: PLC0415
        from .utils.logger import logger as _logger  # noqa: PLC0415
    except Exception as banner_err:  # noqa: BLE001
        with contextlib.suppress(Exception):
            from .utils.logger import logger as _logger  # noqa: PLC0415

            _logger.warning(
                "LeakPro device detection skipped (%s). "
                "Run `python -m leakpro.utils.hardware_setup` for diagnostics.",
                banner_err,
            )
        return

    try:
        device = _get_device()
    except Exception as banner_err:  # noqa: BLE001
        _logger.warning(
            "LeakPro device detection skipped (%s). "
            "Run `python -m leakpro.utils.hardware_setup` for diagnostics.",
            banner_err,
        )
        return

    _logger.info("LeakPro device profile: %s", device.type)
    if device.type == "hpu":
        _logger.info(
            "HPU requires `habana-torch-plugin` and `lightning-habana`; "
            "run `python -m leakpro.utils.hardware_setup --install` for guided setup."
        )


if _os.environ.get("LEAKPRO_QUIET_DEVICE_BANNER") != "1":
    _emit_device_banner()
