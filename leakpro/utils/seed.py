#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Function to seed randomness for different libraries."""
import random

import numpy as np
import torch

from leakpro.utils.device import get_device
from leakpro.utils.logger import logger


def seed_everything(seed: int) -> None:
    """Set the seed for different libraries."""
    torch.manual_seed(seed)
    device = get_device()
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device.type == "hpu":
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
