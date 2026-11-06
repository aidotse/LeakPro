#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""One attack trial: (optionally pre-train) -> victim signal -> attack -> leakage metrics.

This is deliberately *not* a multi-client FL simulation. A single model is trained
centrally for ``epochs`` epochs ("exposure"), one client batch (FedSGD) or one small
client dataset (FedAvg) is drawn from the training data, and the intercepted gradient / update is
inverted. Victim and attacker both use ``model.eval()`` (BatchNorm running stats, no dropout).
"""
from __future__ import annotations

import copy
import time
import warnings
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from examples.gia.common.tabular_data import LOADERS, TabularDataset
from examples.gia.common.tabular_models import PRESETS, build_mlp
from leakpro.attacks.gia_attacks.modular import RunMetadata, TargetInfo, run_attack, simulate_client
from leakpro.attacks.gia_attacks.modular.components.evaluators import TabularEvaluator
from leakpro.attacks.gia_attacks.modular.presets import tabular, tabular_raw_onehot

DEVICE = torch.device("cpu")


def set_device(device: torch.device) -> None:
    """Device for the trials (the examples set it from ``--device``); clears the pretrained-model cache."""
    global DEVICE  # noqa: PLW0603
    if device != DEVICE:
        DEVICE = device
        pretrained.cache_clear()


@lru_cache(maxsize=4)
def get_dataset(name: str) -> TabularDataset:
    return LOADERS[name]()


def _model_spec(model) -> dict:
    return dict(PRESETS[model]) if isinstance(model, str) else dict(model)


def _out_dim(ds: TabularDataset) -> int:
    return ds.num_classes if ds.task_type == "multiclass" else 1


def _loss(ds: TabularDataset, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if ds.task_type == "multiclass":
        return F.cross_entropy(out, y.long())
    if ds.task_type == "binary":
        return F.binary_cross_entropy_with_logits(out.squeeze(-1), y)
    return F.mse_loss(out.squeeze(-1), y)


def utility_metrics(ds: TabularDataset, model: torch.nn.Module) -> dict[str, float]:
    """Test-set utility of the attacked model (ROC-AUC or R2), reported alongside leakage."""
    from sklearn.metrics import r2_score, roc_auc_score

    model.eval()
    with torch.no_grad():
        out = model(ds.x_test.to(DEVICE)).cpu()
    y = ds.y_test.numpy()
    m = {"test_loss": float(_loss(ds, out, ds.y_test))}
    if ds.task_type == "binary":
        p = torch.sigmoid(out.squeeze(-1)).numpy()
        m["test_acc"] = float(((p > 0.5) == (y > 0.5)).mean())
        m["test_roc_auc"] = float(roc_auc_score(y, p))
    elif ds.task_type == "regression":
        m["test_r2"] = float(r2_score(y, out.squeeze(-1).numpy()))
        m["test_mae"] = float(np.abs(y - out.squeeze(-1).numpy()).mean())
    return m


@lru_cache(maxsize=64)
def pretrained(dataset: str, model_key: tuple, seed: int, epochs: int):
    ds = get_dataset(dataset)
    torch.manual_seed(seed)
    model = build_mlp(ds.num_features, _out_dim(ds), **dict(model_key)).to(DEVICE)
    if epochs > 0:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x, y = ds.x_train.to(DEVICE), ds.y_train.to(DEVICE)
        g = torch.Generator().manual_seed(seed)
        model.train()
        for _ in range(epochs):
            for idx in torch.randperm(len(x), generator=g).split(256):
                opt.zero_grad()
                _loss(ds, model(x[idx]), y[idx]).backward()
                opt.step()
    model.eval()
    return model, utility_metrics(ds, model)


def build_attack(cfg: dict, ds: TabularDataset, evaluator: TabularEvaluator):
    common = dict(
        task_type=ds.task_type, label_mode=cfg.get("label_mode", "known"), lr=cfg["lr"], max_iter=cfg["max_iter"],
        evaluator=evaluator, model_mode="eval",
    )
    if cfg.get("protocol", "fedsgd") == "fedavg":
        common.update(compute_mode="updates", num_local_epochs=cfg["local_epochs"], local_batch_size=cfg["batch_size"],
                      meta_lr=cfg.get("meta_lr", 0.01))
    if cfg["variant"] == "raw_onehot":
        return tabular_raw_onehot(**ds.preset_kwargs(), **common)
    if cfg["variant"] == "softmax":  # the default tabular attack (softmax relaxation, cosine + 1e-4 L2, constant lr)
        return tabular(**ds.preset_kwargs(), **{**common, "lr": cfg["lr_softmax"]})
    raise ValueError(f"unknown variant {cfg['variant']!r}")


def run_trial(cfg: dict) -> dict:
    """Run one configuration; returns the config merged with all metrics (flat dict of scalars)."""
    ds = get_dataset(cfg["dataset"])
    spec = _model_spec(cfg["model"])
    model, util = pretrained(cfg["dataset"], tuple(sorted(spec.items())), cfg["seed"], cfg.get("epochs", 0))

    rng = np.random.default_rng(1000 + cfg["seed"])
    n_rows = cfg["client_cap"] if cfg.get("protocol", "fedsgd") == "fedavg" else cfg["batch_size"]
    idx = torch.tensor(rng.choice(len(ds.x_train), size=n_rows, replace=False))
    x, y = ds.x_train[idx].to(DEVICE), ds.y_train[idx].to(DEVICE)
    y_target = y.long() if ds.task_type == "multiclass" else y

    evaluator = TabularEvaluator(
        ds.schema(), std_source=cfg.get("std_source", "unit"), matching=cfg.get("matching", "l1"),
        reference_rows=ds.x_train, random_trials=cfg.get("random_trials", 32), random_seed=cfg["seed"],
    )
    atk = build_attack(cfg, ds, evaluator)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        obs = simulate_client(model, x, TargetInfo(value=y_target), task=copy.deepcopy(atk.task), modality=atk.modality,
                              input_shape=tuple(x.shape))
        target = TargetInfo(value=y_target) if cfg.get("label_mode", "known") == "known" else None
        state, ctx = run_attack(atk, model=model, observation=obs, input_shape=tuple(x.shape), device=DEVICE,
                                target=target, metadata=RunMetadata(seed=cfg["seed"]))
    result = evaluator.evaluate(state.input.detach(), x.detach(), ctx=ctx, state=state)
    row = {**{k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in cfg.items()}, **util,
           **{k: float(v) for k, v in result.metrics.items() if not k.startswith("accuracy/")},
           "attack_seconds": time.time() - t0}
    row["tableak_acc"] = row["reconstruction_accuracy"] / 100.0
    return row
