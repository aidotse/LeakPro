#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""PET optimization campaign on the MIMIC LOS logistic-regression target.

Traces the utility-vs-attack-success frontier for DP-SGD on the LOS binary
classification task. The training loop, attack and utility evaluation all come
from the shared ``PETRecipe`` path in ``leakpro.optimization`` — this file only
declares the recipe (LR model, Adam, BCE loss) and the data splits.

Usage:
    python run_campaign.py --smoke          # 2 configs, quick pipeline check
    python run_campaign.py --n-configs 50   # real Sobol sweep
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))  # mimic_data_handler must be importable for unpickling

from leakpro.optimization import Campaign, PETRecipe, build_campaign_fns, default_dpsgd_space, pareto_front, plot_frontier
from leakpro.utils.logger import logger

N_AUDIT = 2000


def load_splits(seed: int = 0) -> dict:
    """Load the LOS LR dataset and carve disjoint roles from the stored splits.

    train_indices -> target-train + reference pool (disjoint);
    test_indices  -> audit nonmembers + utility eval (disjoint).
    """
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
    # Audit members must be a subset of target_train (the first n_target train
    # indices), and nonmembers plus a non-empty utility split must both fit in
    # the test indices — cap instead of assuming the dataset is large enough.
    n_audit = min(N_AUDIT, n_target, len(test_idx) - 1)
    if n_audit <= 0 or len(test_idx) - n_audit <= 0:
        raise ValueError(f"Dataset too small for the campaign splits: {len(train_idx)} train / "
                         f"{len(test_idx)} test indices.")
    return {
        "x": x,
        "y": y,
        "target_train": train_idx[:n_target],
        "ref_pool": train_idx[n_target:],
        "audit_members": train_idx[:n_audit],  # subset of target_train
        "audit_nonmembers": test_idx[:n_audit],
        "utility_eval": test_idx[n_audit:],
    }


class LRSigmoid(nn.Module):
    """Logistic regression emitting probabilities (matches the LOS example's LR + BCELoss)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


def make_recipe(splits: dict, epochs: int) -> PETRecipe:
    """The LOS training recipe: logistic regression + Adam + BCE."""
    input_dim = splits["x"].shape[1]
    return PETRecipe(
        make_model=lambda config: LRSigmoid(input_dim),
        make_optimizer=lambda params, config: optim.Adam(params, lr=config["learning_rate"]),
        make_loader=lambda indices, config: DataLoader(
            TensorDataset(splits["x"][indices], splits["y"][indices]),
            batch_size=int(config["batch_size"]), shuffle=True,
        ),
        criterion=nn.BCELoss(),
        epochs=epochs,
        output_kind="binary_probs",
    )



def above_chance_auc(floor: float = 0.5):  # noqa: ANN201
    """Utility gate: only attack models that beat chance AUC.

    A binary model at or below 0.5 AUC has not learned; attacking it spends the
    target plus every reference model to measure nothing, and the degenerate
    result can still land on the Pareto front.
    """
    def gate(utility: float, _history) -> bool:  # noqa: ANN001
        if utility <= floor:
            logger.warning(
                f"AUC {utility:.4f} is at or below the {floor:.2f} chance floor: "
                "skipping the attack, this model did not learn."
            )
            return False
        return True

    return gate


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

    # Seed torch too: numpy covers the Sobol draw and the split permutation,
    # but model init, shuffling, Poisson sampling and the DP noise run off
    # torch's global RNG.
    torch.manual_seed(args.seed)
    splits = load_splits(seed=args.seed)
    recipe = make_recipe(splits, args.epochs)
    train_fn, utility_fn, attack_fn = build_campaign_fns(
        recipe, splits, n_refs=args.n_refs, device=args.device, utility_metric="auc")

    campaign = Campaign(
        train_fn, utility_fn, attack_fn,
        knob_space=default_dpsgd_space(),
        output_dir=args.out,
        seed=args.seed,
        utility_gate=above_chance_auc(),
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
