#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Open tabular datasets for the tabular experiments: Adult (binary) and California Housing (regression).

Both come from public sources (OpenML / scikit-learn) and are downloaded on first use. Features are
standardised with the *training split's* statistics (numerics first) followed by one-hot blocks, so
the attacked "client" owns standardised data -- which lets the
evaluator's ``std_source="unit"`` reproduce its ``0.319 * client_std`` tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from leakpro.attacks.gia_attacks.modular.components.tabular import TabularSchema

ADULT_NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
ADULT_CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]


@dataclass
class TabularDataset:
    name: str
    task_type: str                      # "binary" | "regression" | "multiclass"
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    continuous_cols: list[int]
    categorical_col_groups: list[tuple[int, int]]
    col_means: dict[int, float]
    col_stds: dict[int, float]
    feature_names: list[str]
    num_classes: int = 1
    y_stats: tuple[float, float] | None = None   # (mean, std) of the raw regression target

    @property
    def num_features(self) -> int:
        return self.x_train.shape[1]

    def schema(self) -> TabularSchema:
        return TabularSchema(
            num_features=self.num_features,
            continuous_cols=self.continuous_cols,
            categorical_col_groups=self.categorical_col_groups,
            col_means=self.col_means,
            col_stds=self.col_stds,
            feature_names=self.feature_names,
        )

    def preset_kwargs(self) -> dict:
        return dict(
            continuous_cols=self.continuous_cols,
            categorical_col_groups=self.categorical_col_groups,
            col_means=self.col_means,
            col_stds=self.col_stds,
            num_features=self.num_features,
            num_classes=self.num_classes,
            feature_names=self.feature_names,
        )


def _encode(df: pd.DataFrame, numeric: list[str], categorical: list[str], train_idx, test_idx):
    """Standardise numerics on the train split, one-hot the categoricals (numerics first)."""
    num = df[numeric].to_numpy(dtype=np.float64)
    means, stds = num[train_idx].mean(0), num[train_idx].std(0)
    stds[stds == 0] = 1.0
    parts = [(num - means) / stds]
    names = list(numeric)
    groups: list[tuple[int, int]] = []
    start = len(numeric)
    for col in categorical:
        dummies = pd.get_dummies(df[col].astype(str)).to_numpy(dtype=np.float64)
        groups.append((start, dummies.shape[1]))
        start += dummies.shape[1]
        parts.append(dummies)
        names += [f"{col}" for _ in range(dummies.shape[1])]
    x = torch.tensor(np.concatenate(parts, axis=1), dtype=torch.float32)
    cont = list(range(len(numeric)))
    return x, cont, groups, {i: float(means[i]) for i in cont}, {i: float(stds[i]) for i in cont}, names


def _split(n: int, seed: int, test_frac: float = 0.2):
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_frac)
    return perm[n_test:], perm[:n_test]


def load_adult(split_seed: int = 0) -> TabularDataset:
    from sklearn.datasets import fetch_openml

    df = fetch_openml("adult", version=2, as_frame=True, parser="auto").frame
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    y = torch.tensor((df["class"].astype(str).str.contains(">50K")).to_numpy(dtype=np.float32))
    tr, te = _split(len(df), split_seed)
    x, cont, groups, means, stds, names = _encode(df, ADULT_NUMERIC, ADULT_CATEGORICAL, tr, te)
    return TabularDataset("adult", "binary", x[tr], y[tr], x[te], y[te], cont, groups, means, stds, names, num_classes=1)


def load_california(split_seed: int = 0) -> TabularDataset:
    from sklearn.datasets import fetch_california_housing

    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame
    numeric = [c for c in df.columns if c != bunch.target_names[0]]
    tr, te = _split(len(df), split_seed)
    x, cont, groups, means, stds, names = _encode(df, numeric, [], tr, te)
    y_raw = df[bunch.target_names[0]].to_numpy(dtype=np.float64)
    y_mean, y_std = float(y_raw[tr].mean()), float(y_raw[tr].std())
    y = torch.tensor((y_raw - y_mean) / y_std, dtype=torch.float32)
    return TabularDataset("california", "regression", x[tr], y[tr], x[te], y[te], cont, groups, means, stds, names,
                          num_classes=1, y_stats=(y_mean, y_std))


def _synthetic(task_type: str, n_cont: int, cat_sizes: list[int], n: int = 600, split_seed: int = 0) -> TabularDataset:
    """A small random table with a learnable target (``--quick``: no download)."""
    g = torch.Generator().manual_seed(1234)
    cont = torch.randn(n, n_cont, generator=g)
    blocks = [torch.nn.functional.one_hot(torch.randint(0, k, (n,), generator=g), k).float() for k in cat_sizes]
    x = torch.cat([cont, *blocks], dim=1)
    score = cont[:, 0] + 0.5 * cont[:, -1] + sum(b[:, 0] for b in blocks)
    y = (score > score.median()).float() if task_type == "binary" else score
    if task_type == "regression":
        y_mean, y_std = float(y.mean()), float(y.std())
        y = (y - y_mean) / y_std
    tr, te = _split(n, split_seed)
    groups, start = [], n_cont
    for k in cat_sizes:
        groups.append((start, k))
        start += k
    cols = list(range(n_cont))
    names = [f"x{i}" for i in cols] + [f"c{j}" for j, k in enumerate(cat_sizes) for _ in range(k)]
    return TabularDataset("synthetic" if cat_sizes else "synthetic_regression", task_type, x[tr], y[tr], x[te], y[te], cols,
                          groups, {i: 0.0 for i in cols}, {i: 1.0 for i in cols}, names, num_classes=1,
                          y_stats=(y_mean, y_std) if task_type == "regression" else None)


def load_synthetic(split_seed: int = 0) -> TabularDataset:
    """4 continuous + 2 categorical features, binary target."""
    return _synthetic("binary", 4, [3, 3], split_seed=split_seed)


def load_synthetic_regression(split_seed: int = 0) -> TabularDataset:
    """8 continuous features, regression target."""
    return _synthetic("regression", 8, [], split_seed=split_seed)


LOADERS = {"adult": load_adult, "california": load_california, "synthetic": load_synthetic,
           "synthetic_regression": load_synthetic_regression}
