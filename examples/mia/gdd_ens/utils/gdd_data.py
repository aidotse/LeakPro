#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Prepare the GDD_ENS population from the full feature table ``msk_solid_heme_ft.csv``.

Kept separate from the data handler so the handler stays import-light. The feature table is
NOT shipped with the example (GENIE is controlled-access and the file is ~860 MB); the user
provides it — see the README.

We deliberately start from the *full* feature table rather than GDD's ``ft_train_labelled.csv``
output, because the latter is GDD's 80% training split after ``RandomOverSampler`` balancing
(duplicate rows for rare classes) and is unsuitable for a clean membership split. Instead we
reproduce GDD's preprocessing up to (but not including) the train/test split and oversampling:

    * keep only ``Classification_Category == 'train'`` samples (drop low-purity / 'other'),
    * keep one sample per patient (``PATIENT_ID`` = first 9 chars of ``SAMPLE_ID``) so a
      patient never lands in both members and non-members,
    * label-encode the ``Cancer_Type`` target (38 classes),
    * drop the metadata/identifier columns, leaving the raw genomic features,
    * optionally drop features that are constant across the population (GDD drops columns
      constant within its training split; we mirror that on our own population).

No oversampling: LeakPro carves the in/out membership split from this natural population.
Features are used RAW (no scaling), mirroring how GDD fed the feature table to its MLP.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# Identifier / label columns in msk_solid_heme_ft.csv that are not model features.
# (The leading unnamed index column is consumed by index_col=0 on read.)
METADATA_COLS = [
    "SAMPLE_ID", "CANCER_TYPE", "CANCER_TYPE_DETAILED", "Cancer_Type",
    "SAMPLE_TYPE", "PRIMARY_SITE", "METASTATIC_SITE", "Classification_Category",
]
TARGET_COL = "Cancer_Type"
CATEGORY_COL = "Classification_Category"
TRAIN_CATEGORY = "train"
PATIENT_ID_LEN = 9  # GDD derives PATIENT_ID = SAMPLE_ID[:9].


def prepare_gdd_population(
    ft_path: str = "./data/msk_solid_heme_ft.csv",
    drop_constant: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, LabelEncoder]:
    """Build the de-duplicated, non-oversampled GDD population from the full feature table.

    Args:
        ft_path: Path to the GDD full feature table ``msk_solid_heme_ft.csv``.
        drop_constant: If True, drop feature columns that are constant across the population.

    Returns:
        features (float32 tensor, [N, F]), targets (long tensor, [N]), and the fitted
        LabelEncoder (its ``classes_`` map integer labels back to tumor-type names).

    """
    path = Path(ft_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Provide the GDD full feature table msk_solid_heme_ft.csv "
            f"in {path.parent}/ (see the example README)."
        )

    df = pd.read_csv(path, index_col=0)

    # Keep only training-eligible samples (GDD's split_data.py drops low_purity / 'other').
    df = df[df[CATEGORY_COL] == TRAIN_CATEGORY]

    # One sample per patient so members and non-members never share a patient.
    patient_id = df["SAMPLE_ID"].str[:PATIENT_ID_LEN]
    df = df[~patient_id.duplicated()].reset_index(drop=True)

    encoder = LabelEncoder()
    targets = encoder.fit_transform(df[TARGET_COL].to_numpy())

    feature_cols = [c for c in df.columns if c not in set(METADATA_COLS)]
    features_df = df[feature_cols]

    if drop_constant:
        non_constant = features_df.nunique() > 1
        features_df = features_df.loc[:, non_constant]

    features = torch.from_numpy(features_df.to_numpy(dtype=np.float32))
    targets = torch.from_numpy(targets.astype(np.int64))

    # LeakPro's balanced shadow-model assignment (construct_balanced_assignments)
    # requires an even-sized shadow population, and the MIA attacks use the *entire*
    # population as that pool. An odd population therefore raises
    # "Balanced shadow assignments must contain half of the shadow population per
    # model". Drop one sample to make the count even when needed. We drop it from the
    # most frequent class so no class is removed and the label range (and thus the
    # class count the notebook derives from targets.max()) is preserved.
    if features.shape[0] % 2 == 1:
        majority_label = torch.mode(targets).values.item()
        drop_idx = int((targets == majority_label).nonzero(as_tuple=True)[0][0])
        keep = torch.ones(features.shape[0], dtype=torch.bool)
        keep[drop_idx] = False
        features, targets = features[keep], targets[keep]

    return features, targets, encoder
