#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Function to seed randomness for different libraries."""
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set the seed for different libraries.

    Deliberately does not touch the cuDNN flags: forcing
    torch.backends.cudnn.deterministic slows training/inference
    substantially, and benchmark mode is a performance setting unrelated
    to seeding (issue #325). Users who need bit-exact GPU reproducibility
    can set those flags themselves.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
