#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""PET optimization campaign on the MIMIC LOS GRU-D target.

GRU-D is the leakier LOS target (a recurrent net over the raw multivariate time
series, versus the flat logistic-regression target in ``run_campaign.py``), so
its utility-vs-attack frontier has a real privacy axis to trace.

Like the LR and CIFAR campaigns, this file only declares a ``PETRecipe`` (the
GRU-D model, optimizer, loader, loss) and the data splits; the DP-SGD loop,
matched-reference attack and utility evaluation come from the shared core path.

Two GRU-D specifics worth knowing:
- The model outputs a raw logit (BCEWithLogitsLoss), so ``output_kind`` is
  "binary_logits".
- GRU-D contains a custom ``FilterLinear`` and a manual 104-step recurrence.
  Opacus is put on its functorch per-sample-gradient path (``force_functorch``)
  to differentiate it; this is correct but far slower than the LR/CNN targets,
  so keep --n-configs and epochs modest.

Usage:
    python run_campaign_grud.py --smoke          # 2 configs, pipeline check
    python run_campaign_grud.py --n-configs 20   # real (slow) sweep
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim, zeros
from torch.utils.data import DataLoader, TensorDataset

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))  # mimic_data_handler (unpickling) + target_models (GRUD)

from target_models import GRUD  # noqa: E402

from leakpro.optimization import (  # noqa: E402
    Campaign,
    Knob,
    KnobSpace,
    PETRecipe,
    build_campaign_fns,
    pareto_front,
    plot_frontier,
)
from leakpro.utils.logger import logger  # noqa: E402

N_AUDIT = 2000
MAX_PHYSICAL_BATCH = 128


def load_splits(seed: int = 0) -> dict:
    """Load the LOS GRU-D time-series dataset and carve disjoint roles.

    train_indices -> target-train + reference pool (disjoint);
    test_indices  -> audit nonmembers + utility eval (disjoint).
    """
    data_dir = EXAMPLE_DIR / "data" / "GRUD_data"
    with (data_dir / "dataset.pkl").open("rb") as f:
        dataset = pickle.load(f)
    with (data_dir / "indices.pkl").open("rb") as f:
        indices = pickle.load(f)

    x = dataset.data.float()          # [N, time_steps*3, features]
    y = dataset.targets.float().reshape(-1, 1)

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(np.asarray(indices["train_indices"]))
    test_idx = rng.permutation(np.asarray(indices["test_indices"]))

    n_target = len(train_idx) // 2
    # Members must stay inside target_train (the first n_target indices) and the
    # nonmembers must leave a non-empty utility split behind — cap rather than
    # assume the dataset is big enough, as the LR campaign does.
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


def make_recipe(splits: dict, epochs: int, hidden_size: int = 78) -> PETRecipe:
    """The GRU-D training recipe. Init params mirror the LOS example's notebooks.

    ``input_size`` and ``X_mean`` follow the packed [N, time*3, features] layout:
    input_size = time_steps = data.shape[1] // 3, X_mean = zeros(1, features, time_steps).
    """
    x = splits["x"]
    time_steps = x.shape[1] // 3
    features = x.shape[2]
    x_mean = zeros(1, features, time_steps)

    def make_model(config: dict) -> nn.Module:
        return GRUD(
            input_size=time_steps,
            hidden_size=hidden_size,
            X_mean=x_mean,
            batch_size=int(config["batch_size"]),
            bn_flag=False,          # BatchNorm is incompatible with Opacus
            force_functorch=True,   # custom FilterLinear needs the functorch grad sampler
        )

    return PETRecipe(
        make_model=make_model,
        make_optimizer=lambda params, config: optim.Adam(params, lr=config["learning_rate"]),
        make_loader=lambda indices, config: DataLoader(
            TensorDataset(x[indices], splits["y"][indices]),
            batch_size=int(config["batch_size"]), shuffle=True,
        ),
        criterion=nn.BCEWithLogitsLoss(),
        epochs=epochs,
        output_kind="binary_logits",
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


def knob_space() -> KnobSpace:
    """Joint DP-SGD space; batch size kept smaller than LR because GRU-D is memory-heavy."""
    return KnobSpace([
        Knob("noise_multiplier", 0.4, 8.0, log_scale=True),
        Knob("max_grad_norm", 0.1, 10.0, log_scale=True),
        Knob("learning_rate", 1e-4, 1e-2, log_scale=True),
        Knob("batch_size", 32, 256, log_scale=True, integer=True),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-configs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--n-refs", type=int, default=2)
    # No --device flag on purpose: GRUD pins X_mean, the identity matrix and
    # FilterLinear's filter to the auto-detected device at construction, and
    # .to(device) does not move those unregistered attributes — a flag would
    # accept a value it cannot honor.
    parser.add_argument("--out", default="leakpro_output/pet_optimization_grud")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="2 configs, 1 epoch, 1 ref: pipeline check only")
    args = parser.parse_args()

    if args.smoke:
        args.n_configs, args.epochs, args.n_refs = 2, 1, 1
        args.out = args.out + "_smoke"

    # Seed torch too: numpy covers the Sobol draw and the split permutation,
    # but model init, shuffling, Poisson sampling and the DP noise run off
    # torch's global RNG.
    torch.manual_seed(args.seed)
    splits = load_splits(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # see the parser note above
    recipe = make_recipe(splits, args.epochs)
    train_fn, utility_fn, attack_fn = build_campaign_fns(
        recipe, splits, n_refs=args.n_refs, device=device, utility_metric="auc",
        # GRU-D is the memory-heaviest target in the repo: per-sample gradients
        # over the unrolled sequence OOM at the shared 256 default.
        max_physical_batch=MAX_PHYSICAL_BATCH)

    campaign = Campaign(
        train_fn, utility_fn, attack_fn,
        knob_space=knob_space(),
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
