#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Function to seed randomness for different libraries."""
import random

import numpy as np
import torch

from leakpro.utils.device import hpu_is_installed
from leakpro.utils.logger import logger


def seed_everything(seed: int) -> None:
    """Set the seed for different libraries."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Unconditional: several modules still pick CUDA via their own torch.cuda.is_available()
    # check rather than get_device(), so gating this on get_device() could leave CUDA's RNG
    # unseeded while training still runs on it. torch.cuda.manual_seed is a documented no-op
    # when CUDA isn't initialised, so this is harmless when nothing is actually on CUDA.
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Checked directly via hpu_is_installed() rather than get_device(): seeding must not fail
    # just because a card can't be acquired right now (get_device() raises HPUAcquisitionError
    # in that case) — we only need to know the Habana library is present.
    if hpu_is_installed():
        try:
            import habana_frameworks.torch.hpu as hthpu  # type: ignore[import-not-found]  # noqa: PLC0415
            if hasattr(hthpu, "manual_seed_all"):
                hthpu.manual_seed_all(seed)
            elif hasattr(hthpu, "random") and hasattr(hthpu.random, "manual_seed_all"):
                hthpu.random.manual_seed_all(seed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HPU seed could not be set: %s", exc)
    np.random.seed(seed)
    random.seed(seed)
