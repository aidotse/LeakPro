"""Fixtures for Anonymeter's tests."""
# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details..
import os
from typing import Optional

import pandas as pd

TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_adult(*, return_ori: bool, n_samples: Optional[int] = None) -> pd.DataFrame:
    """Fixture for the adult dataset.

    For details see:
    https://archive.ics.uci.edu/ml/datasets/adult

    Parameters
    ----------
    return_ori : bool
        If True - returns "original" samples, otherwise "synthetic" samples
    n_samples : int
        Number of sample records to return.
        If `None` - return all samples.

    Returns
    -------
    df : pd.DataFrame
        Adult dataframe.

    """
    fname = "adults_syn.csv"
    if return_ori:
        fname = "adults_ori.csv"
    return pd.read_csv(os.path.join(TEST_DIR_PATH, "datasets", fname), nrows=n_samples)
