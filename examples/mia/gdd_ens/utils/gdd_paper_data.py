#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Reproduce the GDD-ENS train/test membership split for the ensemble (real-model-style) audit.

Unlike ``utils/gdd_data.py`` (the single-MLP baseline, which dedups to one sample per patient and
drops constant columns on the *whole* population), this loader mirrors ``scripts/split_data.py`` from
the GDD_ENS repo so the member / non-member labels match how the published GDD-ENS model was actually
trained:

    * keep ``Classification_Category == 'train'`` rows (drop low-purity / 'other'),
    * deterministic 80/20 ``train_test_split(..., random_state=0)`` at the SAMPLE level
      (patients are NOT deduplicated before the split, exactly as the paper does),
    * non-members = the 20% test rows MINUS any row whose ``rep_drop`` (PATIENT_ID + Cancer_Type)
      also appears in the train split (the paper's test-set de-duplication),
    * features = all columns minus the metadata/id columns, then drop columns constant in the
      *training* split (the paper's ``remove=True``),
    * NO oversampling — LeakPro/the handler train on the natural in-split.

Because the split is fully deterministic, the membership ground truth is reproducible without the
paper's Supplementary Table S1.

Returns one combined population (members first, then non-members) plus the index ranges, so the
notebook can record ``train_indices`` (members) / ``test_indices`` (non-members) in the target
metadata and let LeakPro evaluate the attack against the true membership.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Non-feature columns. SAMPLE_ID/Cancer_Type/Classification_Category come from the feature table;
# PATIENT_ID/rep_drop are derived below (mirroring split_data.py).
METADATA_COLS = [
    "SAMPLE_ID", "PATIENT_ID", "CANCER_TYPE", "CANCER_TYPE_DETAILED", "SAMPLE_TYPE",
    "PRIMARY_SITE", "METASTATIC_SITE", "Cancer_Type", "Classification_Category", "rep_drop",
]
TARGET_COL = "Cancer_Type"
PATIENT_ID_LEN = 9  # GDD derives PATIENT_ID = SAMPLE_ID[:9].


@dataclass
class GddPaperPopulation:
    """Combined member+non-member population with the ground-truth membership index ranges."""

    features: torch.Tensor          # float32 [N, F], members first then non-members
    targets: torch.Tensor           # long [N]
    member_indices: np.ndarray      # indices into features that were in the model's training set
    nonmember_indices: np.ndarray   # held-out (non-member) indices
    encoder: LabelEncoder           # encoder.classes_ maps integer labels -> tumor-type names
    feature_columns: list[str]      # kept feature column names, in order


def prepare_gdd_paper_population(
    ft_path: str = "./data/msk_solid_heme_ft.csv",
    test_size: int = 20,
    drop_constant: bool = True,
) -> GddPaperPopulation:
    """Build the deterministic member/non-member population mirroring GDD_ENS split_data.py.

    Args:
        ft_path: Path to the GDD full feature table ``msk_solid_heme_ft.csv``.
        test_size: Held-out percentage (paper uses 20), passed to train_test_split as test_size/100.
        drop_constant: Drop feature columns constant within the training split (paper's remove=True).

    Returns:
        A :class:`GddPaperPopulation`.
    """
    path = Path(ft_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Provide the GDD full feature table msk_solid_heme_ft.csv "
            f"in {path.parent}/ (see the example README)."
        )

    data = pd.read_csv(path, index_col=0)
    data = data.assign(PATIENT_ID=data["SAMPLE_ID"].str[:PATIENT_ID_LEN])
    data = data.assign(rep_drop=data["PATIENT_ID"] + data["Cancer_Type"])
    data = data[data["Classification_Category"] == "train"]

    labels = data[TARGET_COL]
    # Deterministic split (random_state=0) at the sample level — exactly split_data.py.
    train_df, test_df, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size / 100, random_state=0)

    # Paper's test de-duplication: drop test rows sharing a (patient, type) with the train split.
    keep_test = ~test_df["rep_drop"].isin(train_df["rep_drop"].values)
    test_df, test_labels = test_df[keep_test], test_labels[keep_test]

    # Feature columns = everything except metadata; constant-in-train columns dropped (remove=True).
    feature_cols = [c for c in data.columns if c not in set(METADATA_COLS)]
    if drop_constant:
        nunique_train = train_df[feature_cols].nunique()
        feature_cols = [c for c in feature_cols if nunique_train[c] > 1]

    members_X = train_df[feature_cols].to_numpy(dtype=np.float32)
    nonmembers_X = test_df[feature_cols].to_numpy(dtype=np.float32)
    features = np.concatenate([members_X, nonmembers_X], axis=0)

    # Encode targets over the combined population so every label seen anywhere is mapped.
    encoder = LabelEncoder()
    all_labels = pd.concat([train_labels, test_labels]).to_numpy()
    targets = encoder.fit_transform(all_labels).astype(np.int64)

    n_members = members_X.shape[0]
    member_indices = np.arange(n_members)
    nonmember_indices = np.arange(n_members, features.shape[0])

    # LeakPro's construct_balanced_assignments needs an even population (LiRA/RMIA pass the whole
    # population as the shadow pool). Drop one non-member from the most frequent class if odd, so all
    # classes and the label range are preserved.
    if features.shape[0] % 2 == 1:
        nonmember_targets = targets[nonmember_indices]
        majority_label = np.bincount(nonmember_targets).argmax()
        drop_pos = nonmember_indices[nonmember_targets == majority_label][0]
        keep = np.ones(features.shape[0], dtype=bool)
        keep[drop_pos] = False
        features, targets = features[keep], targets[keep]
        n_members = int(keep[:n_members].sum())  # unchanged (we dropped a non-member), but recompute defensively
        member_indices = np.arange(n_members)
        nonmember_indices = np.arange(n_members, features.shape[0])

    return GddPaperPopulation(
        features=torch.from_numpy(features),
        targets=torch.from_numpy(targets),
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        encoder=encoder,
        feature_columns=feature_cols,
    )
