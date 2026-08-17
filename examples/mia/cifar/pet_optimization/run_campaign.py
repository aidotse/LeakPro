#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""PET optimization campaign on CIFAR-10 with a small CNN.

The leaky-by-design counterpart of the LOS campaign: a small convolutional
network trained on a 15k subset of CIFAR-10 memorizes visibly, so the
utility-vs-attack frontier has a real privacy axis to trace.

The training loop, attack and utility evaluation all come from the shared
``PETRecipe`` path in ``leakpro.optimization`` — this file only declares the
recipe (model, optimizer, loader, loss) and the data splits.

Usage:
    python run_campaign.py --smoke          # 2 configs, pipeline check
    python run_campaign.py --n-configs 50   # real sweep
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
sys.path.insert(0, str(EXAMPLE_DIR))  # cifar_data_handler must be importable for unpickling

from leakpro.optimization import (
    Campaign,
    Knob,
    KnobSpace,
    PETRecipe,
    build_campaign_fns,
    pareto_front,
    plot_frontier,
)
from leakpro.utils.logger import logger

N_TARGET_TRAIN = 15000
N_AUDIT = 2000
N_UTILITY_EVAL = 5000


class SmallCNN(nn.Module):
    """Opacus-compatible CIFAR CNN (GroupNorm, no BatchNorm). Trains in ~1 min, memorizes visibly."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_splits(seed: int = 0, audit_size: int = N_AUDIT) -> dict:
    """Load CIFAR-10 (60k) and carve disjoint roles: target train, reference pool, audit, utility.

    ``audit_size`` enlarges the audit sets for the validation pass (tail FPRs need
    far more nonmembers than the loop does); the loop keeps the default.
    """
    with (EXAMPLE_DIR / "data" / "cifar10.pkl").open("rb") as f:
        dataset = pickle.load(f)

    x = ((dataset.data - dataset.mean) / dataset.std).float()
    y = dataset.targets.long()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    target_train = order[:N_TARGET_TRAIN]
    ref_pool = order[N_TARGET_TRAIN:2 * N_TARGET_TRAIN + 10000]
    rest = order[2 * N_TARGET_TRAIN + 10000:]

    # Nonmembers must not overlap the utility split; members are drawn from the
    # target's own training set, so the cap is the training-set size.
    n_audit = min(audit_size, len(target_train), len(rest) - N_UTILITY_EVAL)
    if n_audit < audit_size:
        logger.warning(f"Audit size capped at {n_audit} (requested {audit_size}) by the available disjoint data.")
    return {
        "x": x,
        "y": y,
        "target_train": target_train,
        "ref_pool": ref_pool,
        "audit_members": target_train[:N_AUDIT],
        "audit_nonmembers": rest[:N_AUDIT],
        "audit_members_large": target_train[:n_audit],
        "audit_nonmembers_large": rest[:n_audit],
        "utility_eval": rest[n_audit:n_audit + N_UTILITY_EVAL],
    }


def make_recipe(splits: dict, epochs: int) -> PETRecipe:
    """The CIFAR training recipe: SmallCNN + SGD(momentum) + CrossEntropy."""
    return PETRecipe(
        make_model=lambda config: SmallCNN(),
        make_optimizer=lambda params, config: optim.SGD(params, lr=config["learning_rate"], momentum=0.9),
        make_loader=lambda indices, config: DataLoader(
            TensorDataset(splits["x"][indices], splits["y"][indices]),
            batch_size=int(config["batch_size"]), shuffle=True,
        ),
        criterion=nn.CrossEntropyLoss(),
        epochs=epochs,
        output_kind="logits",
    )


def knob_space(include_nonprivate: bool) -> KnobSpace:
    """Joint DP-SGD space; optionally let noise reach 0 to anchor the leaky end of the frontier."""
    noise_low = 0.0 if include_nonprivate else 0.4
    return KnobSpace([
        Knob("noise_multiplier", noise_low, 4.0, log_scale=not include_nonprivate),
        Knob("max_grad_norm", 0.1, 10.0, log_scale=True),
        Knob("learning_rate", 1e-3, 0.5, log_scale=True),
        Knob("batch_size", 64, 1024, log_scale=True, integer=True),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-configs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-refs", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="leakpro_output/pet_optimization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-nonprivate", action="store_true",
                        help="let noise_multiplier reach 0 (linear scale) to anchor the leaky end")
    parser.add_argument("--smoke", action="store_true", help="2 configs, 2 epochs, 1 ref: pipeline check only")
    args = parser.parse_args()

    if args.smoke:
        args.n_configs, args.epochs, args.n_refs = 2, 2, 1
        args.out = args.out + "_smoke"

    splits = load_splits(seed=args.seed)
    recipe = make_recipe(splits, args.epochs)
    train_fn, utility_fn, attack_fn = build_campaign_fns(
        recipe, splits, n_refs=args.n_refs, device=args.device, utility_metric="accuracy")

    campaign = Campaign(
        train_fn, utility_fn, attack_fn,
        knob_space=knob_space(args.include_nonprivate),
        output_dir=args.out,
        seed=args.seed,
    )
    start = time.time()
    records = campaign.run(args.n_configs)
    logger.info(f"Campaign done: {len(records)} configs in {time.time() - start:.0f}s.")

    front = pareto_front(records)
    logger.info(f"Pareto front ({len(front)} points):")
    for r in front:
        logger.info(f"  utility={r['utility']:.4f}  TPR@1%={r['attack_tpr']:.4f}  "
                    f"eps={r.get('epsilon', float('nan')):.2f}  config={r['config']}")
    plot_path = plot_frontier(records, Path(args.out) / "frontier.png")
    logger.info(f"Frontier plot: {plot_path}")


if __name__ == "__main__":
    main()
