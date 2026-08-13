#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""PET optimization campaign on the MIMIC LOS logistic-regression target.

Traces the utility-vs-attack-success frontier for DP-SGD on the LOS binary
classification task, using the leakpro.optimization core module.

- Knobs (joint search, per the frontier plan): noise multiplier, clipping
  norm, learning rate, batch size. Epochs are fixed.
- Utility: AUC on a held-out test split.
- Attack: matched-reference MIA (RMIA-style calibration with 1-2 reference
  models trained with the SAME configuration as the candidate — full mimicry).
  Signal: logit-scaled confidence of the target minus the reference mean.
- Loop metric: TPR @ FPR = 1% (powered proxy; tail FPRs are validation-only).

Usage:
    python run_campaign.py --smoke          # 2 configs, quick pipeline check
    python run_campaign.py --n-configs 50   # real Sobol sweep
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from opacus import PrivacyEngine
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))  # mimic_data_handler must be importable for unpickling

from leakpro.optimization import AttackScores, Campaign, default_dpsgd_space, pareto_front, plot_frontier
from leakpro.utils.logger import logger

DELTA = 1e-5


def load_splits(seed: int = 0) -> dict:
    """Load the LOS LR dataset and carve person-disjoint roles from the stored splits.

    train_indices -> target-train + reference pool (disjoint);
    test_indices  -> audit nonmembers + utility eval (disjoint).
    """
    import pickle

    data_dir = EXAMPLE_DIR / "data" / "LR_data"
    with (data_dir / "dataset.pkl").open("rb") as f:
        dataset = pickle.load(f)
    with (data_dir / "indices.pkl").open("rb") as f:
        indices = pickle.load(f)

    x = dataset.data.float()
    y = dataset.targets.float().reshape(-1, 1)

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(np.asarray(indices["train_indices"]))
    test_idx = rng.permutation(np.asarray(indices["test_indices"]))

    n_target = len(train_idx) // 2
    n_audit = 2000
    return {
        "x": x,
        "y": y,
        "target_train": train_idx[:n_target],
        "ref_pool": train_idx[n_target:],
        "audit_members": train_idx[:n_audit],  # subset of target_train
        "audit_nonmembers": test_idx[:n_audit],
        "utility_eval": test_idx[n_audit:],
    }


def train_dpsgd_lr(config: dict, splits: dict, train_indices: np.ndarray,
                   epochs: int, device: str) -> nn.Module:
    """Train one logistic-regression model with DP-SGD under the sampled config."""
    from target_models import LR

    x, y = splits["x"], splits["y"]
    loader = DataLoader(
        TensorDataset(x[train_indices], y[train_indices]),
        batch_size=int(config["batch_size"]), shuffle=True,
    )
    model = LR(input_dim=x.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCELoss()

    engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=config["noise_multiplier"],
        max_grad_norm=config["max_grad_norm"],
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

    epsilon = engine.get_epsilon(delta=DELTA)
    logger.info(f"Trained DP-SGD LR: formal epsilon = {epsilon:.2f} (delta = {DELTA}).")
    model.campaign_extras = {"epsilon": epsilon, "delta": DELTA}  # recorded next to the attack result
    return model.eval()


@torch.no_grad()
def _confidence_logits(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                       device: str, batch: int = 4096) -> np.ndarray:
    """Logit-scaled confidence in the TRUE label, phi = log p_true - log(1 - p_true)."""
    outs = []
    for i in range(0, len(x), batch):
        p = model(x[i:i + batch].to(device)).cpu().clamp(1e-6, 1 - 1e-6)
        p_true = torch.where(y[i:i + batch] > 0.5, p, 1 - p)
        outs.append(torch.log(p_true) - torch.log1p(-p_true))
    return torch.cat(outs).numpy().ravel()


def make_fns(splits: dict, epochs: int, n_refs: int, device: str, ref_seed: int = 1) -> tuple:
    """Build the campaign's (train, utility, attack) callables."""

    def train_fn(config: dict) -> nn.Module:
        return train_dpsgd_lr(config, splits, splits["target_train"], epochs, device)

    def utility_fn(model: nn.Module) -> float:
        idx = splits["utility_eval"]
        phi = model(splits["x"][idx].to(device)).detach().cpu().numpy().ravel()
        return float(roc_auc_score(splits["y"][idx].numpy().ravel(), phi))

    def attack_fn(model: nn.Module, config: dict) -> AttackScores:
        # Full mimicry: references use the SAME config (and epochs) as the candidate,
        # trained on data disjoint from the target's training set.
        rng = np.random.default_rng(ref_seed)
        x, y = splits["x"], splits["y"]
        members, nonmembers = splits["audit_members"], splits["audit_nonmembers"]

        ref_models = []
        pool = splits["ref_pool"]
        for _ in range(n_refs):
            sub = rng.choice(pool, size=min(len(splits["target_train"]), len(pool)), replace=False)
            ref_models.append(train_dpsgd_lr(config, splits, sub, epochs, device))

        def calibrated(idx: np.ndarray) -> np.ndarray:
            target_phi = _confidence_logits(model, x[idx], y[idx], device)
            ref_phi = np.mean([_confidence_logits(r, x[idx], y[idx], device) for r in ref_models], axis=0)
            return target_phi - ref_phi

        return AttackScores(member_scores=calibrated(members), nonmember_scores=calibrated(nonmembers))

    return train_fn, utility_fn, attack_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-configs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-refs", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="leakpro_output/pet_optimization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="2 configs, 2 epochs, 1 ref: pipeline check only")
    args = parser.parse_args()

    if args.smoke:
        args.n_configs, args.epochs, args.n_refs = 2, 2, 1
        args.out = args.out + "_smoke"

    splits = load_splits(seed=args.seed)
    train_fn, utility_fn, attack_fn = make_fns(splits, args.epochs, args.n_refs, args.device)

    campaign = Campaign(
        train_fn, utility_fn, attack_fn,
        knob_space=default_dpsgd_space(),
        output_dir=args.out,
        seed=args.seed,
    )
    start = time.time()
    records = campaign.run(args.n_configs)
    logger.info(f"Campaign done: {len(records)} configs in {time.time() - start:.0f}s.")

    front = pareto_front(records)
    logger.info(f"Pareto front ({len(front)} points):")
    for r in front:
        logger.info(f"  utility={r['utility']:.4f}  TPR@1%={r['attack_tpr']:.4f}  config={r['config']}")
    plot_path = plot_frontier(records, Path(args.out) / "frontier.png")
    logger.info(f"Frontier plot: {plot_path}")


if __name__ == "__main__":
    main()
