#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""PET optimization campaign on CIFAR-10 with a small CNN.

The leaky-by-design counterpart of the LOS campaign: a small convolutional
network trained on a 15k subset of CIFAR-10 memorizes visibly, so the
utility-vs-attack frontier has a real privacy axis to trace. Same design as
the LOS example:

- Joint DP-SGD knobs {noise multiplier, clip norm, lr, batch size}; noise may
  be swept down to (near) zero via --include-nonprivate to anchor the leaky end.
- Utility: test accuracy.
- Attack: matched-reference MIA, references trained with the candidate's
  config on disjoint data (full mimicry).
- Loop metric: TPR @ FPR = 1%.

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
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))  # cifar_data_handler must be importable for unpickling

from leakpro.optimization import AttackScores, Campaign, Knob, KnobSpace, pareto_front, plot_frontier
from leakpro.utils.logger import logger

DELTA = 1e-5
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


def train_dpsgd_cnn(config: dict, splits: dict, train_indices: np.ndarray,
                    epochs: int, device: str) -> nn.Module:
    """Train one SmallCNN with DP-SGD under the sampled config (noise 0 = non-private SGD baseline)."""
    x, y = splits["x"], splits["y"]
    loader = DataLoader(
        TensorDataset(x[train_indices], y[train_indices]),
        batch_size=int(config["batch_size"]), shuffle=True,
    )
    model = SmallCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epsilon = float("inf")
    if config["noise_multiplier"] > 0:
        engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=config["noise_multiplier"],
            max_grad_norm=config["max_grad_norm"],
        )

    def run_epochs(epoch_loader) -> None:
        for _ in range(epochs):
            for xb, yb in epoch_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()

    model.train()
    if config["noise_multiplier"] > 0:
        # Cap the physical batch: per-example gradients cost batch x params memory.
        # The sampled (logical) batch size, and hence the accounting, is unchanged.
        with BatchMemoryManager(data_loader=loader, max_physical_batch_size=256,
                                optimizer=optimizer) as mem_loader:
            run_epochs(mem_loader)
        epsilon = engine.get_epsilon(delta=DELTA)
    else:
        run_epochs(loader)
    logger.info(f"Trained CNN: formal epsilon = {epsilon:.2f} (delta = {DELTA}).")
    model.campaign_extras = {"epsilon": epsilon, "delta": DELTA}
    return model.eval()


@torch.no_grad()
def _confidence_logits(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                       device: str, batch: int = 1024) -> np.ndarray:
    """Logit-scaled confidence in the TRUE class: phi = log p_y - log(1 - p_y) (Carlini scaling)."""
    outs = []
    for i in range(0, len(x), batch):
        probs = torch.softmax(model(x[i:i + batch].to(device)), dim=1).cpu().clamp(1e-6, 1 - 1e-6)
        p_true = probs.gather(1, y[i:i + batch].reshape(-1, 1)).squeeze(1)
        outs.append(torch.log(p_true) - torch.log1p(-p_true))
    return torch.cat(outs).numpy()


def make_fns(splits: dict, epochs: int, n_refs: int, device: str, ref_seed: int = 1) -> tuple:
    """Build the campaign's (train, utility, attack) callables."""

    def train_fn(config: dict) -> nn.Module:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()  # models from the previous config are gone; release their cache
        return train_dpsgd_cnn(config, splits, splits["target_train"], epochs, device)

    @torch.no_grad()
    def utility_fn(model: nn.Module) -> float:
        idx = splits["utility_eval"]
        correct, total = 0, 0
        for i in range(0, len(idx), 1024):
            batch = idx[i:i + 1024]
            pred = model(splits["x"][batch].to(device)).argmax(dim=1).cpu()
            correct += int((pred == splits["y"][batch]).sum())
            total += len(batch)
        return correct / total

    def attack_fn(model: nn.Module, config: dict) -> AttackScores:
        # Full mimicry: references share the candidate's config, on disjoint data.
        rng = np.random.default_rng(ref_seed)
        x, y = splits["x"], splits["y"]
        members, nonmembers = splits["audit_members"], splits["audit_nonmembers"]

        # Train references one at a time and free each after scoring (GPU memory).
        ref_phi_members = np.zeros(len(members))
        ref_phi_nonmembers = np.zeros(len(nonmembers))
        for _ in range(n_refs):
            sub = rng.choice(splits["ref_pool"], size=N_TARGET_TRAIN, replace=False)
            ref = train_dpsgd_cnn(config, splits, sub, epochs, device)
            ref_phi_members += _confidence_logits(ref, x[members], y[members], device) / n_refs
            ref_phi_nonmembers += _confidence_logits(ref, x[nonmembers], y[nonmembers], device) / n_refs
            del ref
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        return AttackScores(
            member_scores=_confidence_logits(model, x[members], y[members], device) - ref_phi_members,
            nonmember_scores=_confidence_logits(model, x[nonmembers], y[nonmembers], device) - ref_phi_nonmembers,
        )

    return train_fn, utility_fn, attack_fn


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
    train_fn, utility_fn, attack_fn = make_fns(splits, args.epochs, args.n_refs, args.device)

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
