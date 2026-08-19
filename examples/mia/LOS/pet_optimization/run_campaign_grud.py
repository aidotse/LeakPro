#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""PET optimization campaign on the MIMIC LOS GRU-D target.

The leakier LOS counterpart to run_campaign.py (logistic regression): GRU-D is
a recurrent net over the raw multivariate time series, so its
utility-vs-attack-success frontier has a real privacy axis to trace.

Same design as the LR campaign — joint DP-SGD knob search, matched-reference
MIA (full mimicry), AUC utility, TPR@1% loop metric — with two GRU-D specifics:

- GRU-D outputs a raw logit (BCEWithLogitsLoss); the membership signal is the
  signed logit (phi = logit for members of the true class, negated otherwise).
- GRU-D contains a custom FilterLinear and a manual 104-step recurrence, so
  Opacus is put on its functorch per-sample-gradient path (force_functorch).
  Correct but slower than LR/CNN targets: keep --n-configs and epochs modest.

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
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import roc_auc_score
from torch import nn, optim, zeros
from torch.utils.data import DataLoader, TensorDataset

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))  # mimic_data_handler (unpickling) + target_models (GRUD)

from target_models import GRUD  # noqa: E402

from leakpro.optimization import AttackScores, Campaign, Knob, KnobSpace, pareto_front, plot_frontier  # noqa: E402
from leakpro.utils.logger import logger  # noqa: E402

DELTA = 1e-5
N_AUDIT = 2000
HIDDEN_SIZE = 78


def load_splits(seed: int = 0) -> dict:
    """Load the LOS GRU-D time-series dataset and carve person-disjoint roles."""
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
    return {
        "x": x,
        "y": y,
        "target_train": train_idx[:n_target],
        "ref_pool": train_idx[n_target:],
        "audit_members": train_idx[:N_AUDIT],  # subset of target_train
        "audit_nonmembers": test_idx[:N_AUDIT],
        "utility_eval": test_idx[N_AUDIT:],
    }


def _build_grud(x: torch.Tensor, batch_size: int) -> nn.Module:
    """Instantiate GRU-D; init params mirror the LOS example notebooks."""
    time_steps = x.shape[1] // 3
    features = x.shape[2]
    return GRUD(
        input_size=time_steps,
        hidden_size=HIDDEN_SIZE,
        X_mean=zeros(1, features, time_steps),
        batch_size=batch_size,
        bn_flag=False,          # BatchNorm is incompatible with Opacus
        force_functorch=True,   # custom FilterLinear needs the functorch grad sampler
    )


def train_dpsgd_grud(config: dict, splits: dict, train_indices: np.ndarray,
                     epochs: int, device: str) -> nn.Module:
    """Train one GRU-D model with DP-SGD under the sampled config."""
    x, y = splits["x"], splits["y"]
    loader = DataLoader(
        TensorDataset(x[train_indices], y[train_indices]),
        batch_size=int(config["batch_size"]), shuffle=True,
    )
    model = _build_grud(x, int(config["batch_size"])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=config["noise_multiplier"],
        max_grad_norm=config["max_grad_norm"],
    )

    model.train()
    with BatchMemoryManager(data_loader=loader, max_physical_batch_size=128,
                            optimizer=optimizer) as mem_loader:
        for _ in range(epochs):
            for xb, yb in mem_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()

    epsilon = engine.get_epsilon(delta=DELTA)
    logger.info(f"Trained DP-SGD GRU-D: formal epsilon = {epsilon:.2f} (delta = {DELTA}).")
    model.campaign_extras = {"epsilon": epsilon, "delta": DELTA}
    return model.eval()


@torch.no_grad()
def _confidence_logits(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                       device: str, batch: int = 1024) -> np.ndarray:
    """Signed logit as the membership signal: phi = logit for the true class, negated otherwise."""
    outs = []
    for i in range(0, len(x), batch):
        logit = model(x[i:i + batch].to(device)).cpu().reshape(-1)
        y_batch = y[i:i + batch].reshape(-1)
        outs.append(torch.where(y_batch > 0.5, logit, -logit))
    return torch.cat(outs).numpy()


def make_fns(splits: dict, epochs: int, n_refs: int, device: str, ref_seed: int = 1) -> tuple:
    """Build the campaign's (train, utility, attack) callables."""

    def train_fn(config: dict) -> nn.Module:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return train_dpsgd_grud(config, splits, splits["target_train"], epochs, device)

    @torch.no_grad()
    def utility_fn(model: nn.Module) -> float:
        idx = splits["utility_eval"]
        scores = []
        for i in range(0, len(idx), 1024):
            scores.append(model(splits["x"][idx[i:i + 1024]].to(device)).cpu().reshape(-1))
        return float(roc_auc_score(splits["y"][idx].numpy().ravel(), torch.cat(scores).numpy()))

    def attack_fn(model: nn.Module, config: dict) -> AttackScores:
        # Full mimicry: references share the candidate's config, on disjoint data.
        rng = np.random.default_rng(ref_seed)
        x, y = splits["x"], splits["y"]
        members, nonmembers = splits["audit_members"], splits["audit_nonmembers"]

        ref_phi_m = np.zeros(len(members))
        ref_phi_n = np.zeros(len(nonmembers))
        for _ in range(n_refs):
            sub = rng.choice(splits["ref_pool"], size=len(splits["target_train"]), replace=False)
            ref = train_dpsgd_grud(config, splits, sub, epochs, device)
            ref_phi_m += _confidence_logits(ref, x[members], y[members], device) / n_refs
            ref_phi_n += _confidence_logits(ref, x[nonmembers], y[nonmembers], device) / n_refs
            del ref
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        return AttackScores(
            member_scores=_confidence_logits(model, x[members], y[members], device) - ref_phi_m,
            nonmember_scores=_confidence_logits(model, x[nonmembers], y[nonmembers], device) - ref_phi_n,
        )

    return train_fn, utility_fn, attack_fn



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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_fn, utility_fn, attack_fn = make_fns(splits, args.epochs, args.n_refs, device)

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
