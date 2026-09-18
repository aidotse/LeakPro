#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""DP-SGD privacy-utility optimization on CIFAR-10 with a small CNN.

Traces the privacy-utility frontier for DP-SGD by *optimizing*, not sweeping:
Optuna proposes each next configuration of the DP-SGD knobs from the
(utility, attack-TPR) results seen so far. Each configuration is evaluated by

  1. training a target CNN under the knobs (DP-SGD, noise set directly), and
  2. auditing it with LeakPro's own RMIA attack (``leakpro.attacks.mia_attacks.rmia``)
     run through the ordinary LeakPro pipeline — the RMIA shadow models are
     retrained under the candidate's exact configuration, so they mimic it.

Objectives: maximize the configured utility metric, minimize RMIA TPR at the
configured proxy FPR. Models that fail the utility gate ("did it learn at all?")
are pruned: no attack is run and they can never sit on the frontier.

All settings live in ``dpsgd_optimization.yaml`` (schema:
``leakpro.schemas.PrivacyUtilityConfig``). The per-trial ``audit.yaml`` files are
generated — they are output, not user input.

Usage:
    python run_optimization.py --smoke           # tiny end-to-end pipeline check
    python run_optimization.py                   # dpsgd_optimization.yaml as-is
    python run_optimization.py --config my_settings.yaml
"""

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # make dp_handler importable (also for unpickling)
sys.path.insert(0, str(EXAMPLE_DIR))  # cifar_data_handler is needed to unpickle cifar10.pkl

from dp_handler import CifarDPHandler, SmallCNN, dp_train  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

from leakpro import LeakPro  # noqa: E402
from leakpro.optimization import (  # noqa: E402
    TABULATED_FPRS,
    KnobSpace,
    ObjectiveResult,
    optimize,
    pareto_trials,
    plot_frontier,
    resolved_proxy_tpr,
    run_rmia_audit,
)
from leakpro.schemas import PrivacyUtilityConfig  # noqa: E402
from leakpro.utils.logger import logger  # noqa: E402

NUM_CLASSES = 10


# --------------------------------------------------------------------------- #
# Utility metric + gate
# --------------------------------------------------------------------------- #
def chance_level(metric: str, num_classes: int) -> float:
    """The metric's value for a model that learned nothing.

    Plain accuracy's chance level (1/num_classes) is only meaningful on
    class-balanced data — on imbalanced data an always-majority model scores the
    majority frequency; use balanced_accuracy or auc there instead.
    """
    return 0.5 if metric == "auc" else 1.0 / num_classes


@torch.no_grad()
def evaluate_utility(model: nn.Module, loader: DataLoader, metric: str, device: str) -> float:
    """Compute the configured utility metric on a held-out split."""
    model = model.to(device)
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(softmax(model(xb.to(device)), dim=1).cpu().numpy())
        labels.append(yb.numpy().ravel())
    p = np.concatenate(probs)
    y = np.concatenate(labels)
    if metric == "accuracy":
        return float((p.argmax(1) == y).mean())
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y, p.argmax(1)))
    if metric == "auc":
        return float(roc_auc_score(y, p, multi_class="ovr"))
    raise ValueError(f"Unknown utility metric: {metric}")


# --------------------------------------------------------------------------- #
# Campaign plumbing
# --------------------------------------------------------------------------- #
def build_population(cfg: PrivacyUtilityConfig, out_dir: Path) -> dict:
    """Normalize a CIFAR-10 subset, pickle it as the audit population, carve index roles.

    The population is stored as a ``CifarDPHandler.UserDataset`` (normalized data),
    which is what LeakPro loads via ``target.data_path``. Members are the target's
    training indices; nonmembers/utility come from a disjoint test split. The whole
    population is available to RMIA for training shadow models.
    """
    s = cfg.splits
    if s.n_target + s.n_test > s.pop_size:
        raise ValueError(f"pop_size={s.pop_size} too small for n_target={s.n_target} + n_test={s.n_test}.")
    # RMIA's balanced shadow sampling trains every reference model on
    # len(population) // 2 points. Mimicry of the target's DP-SGD regime
    # (sampling rate q = B/N and steps per epoch) therefore holds only when
    # pop_size = 2 * n_target — enforce it rather than silently calibrating
    # against references trained under a different noise-per-example regime.
    if s.pop_size != 2 * s.n_target:
        raise ValueError(
            f"pop_size={s.pop_size} must be exactly 2 * n_target ({2 * s.n_target}): RMIA's balanced "
            "shadow sampling trains each reference on pop_size // 2 points, and only this coupling "
            "makes the references' training-set size match the target's (full mimicry)."
        )

    with (EXAMPLE_DIR / "data" / "cifar10.pkl").open("rb") as f:
        dataset = pickle.load(f)
    x = ((dataset.data - dataset.mean) / dataset.std).float()
    y = dataset.targets.long()

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(x))[:s.pop_size]

    population = CifarDPHandler.UserDataset(x[order], y[order])
    pop_path = out_dir / "population.pkl"
    with pop_path.open("wb") as f:
        pickle.dump(population, f)

    return {
        "path": str(pop_path),
        "train_indices": list(range(s.n_target)),
        "test_indices": list(range(s.n_target, s.n_target + s.n_test)),
    }


def _config_dir(out_dir: Path, config: dict) -> Path:
    """Stable per-configuration directory (so resume reuses a config's artifacts)."""
    key = json.dumps({k: round(float(v), 8) for k, v in sorted(config.items())})
    tag = hashlib.sha1(key.encode()).hexdigest()[:10]  # noqa: S324 — not security, just a dir name
    return out_dir / "trials" / f"config_{tag}"


def _write_audit_yaml(path: Path, pop_path: str, target_folder: Path,  # noqa: PLR0913
                      dpsgd_path: Path, audit_dir: Path, rmia_config: dict, seed: int) -> None:
    """Write a per-config audit config that runs RMIA against the saved target."""
    module_path = str(Path(__file__).resolve().parent / "dp_handler.py")
    config = {
        "audit": {
            "random_seed": seed,
            "attack_list": [{"attack": "rmia", **rmia_config}],
            "output_dir": str(audit_dir),
            "attack_type": "mia",
            "data_modality": "image",
        },
        "target": {
            "module_path": module_path,
            "model_class": "SmallCNN",
            "target_folder": str(target_folder),
            "data_path": pop_path,
            "dpsgd_path": str(dpsgd_path),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f)


def make_objective(cfg: PrivacyUtilityConfig, pop: dict, out_dir: Path, device: str):  # noqa: ANN201
    """Build the optimization objective: train a target under a config, gate it, audit with real RMIA."""
    with open(pop["path"], "rb") as f:
        population = pickle.load(f)
    x = population.data
    y = population.targets
    train_idx = np.array(pop["train_indices"])
    test_idx = np.array(pop["test_indices"])
    gate = cfg.utility.gate if cfg.utility.gate is not None else chance_level(cfg.utility.metric, NUM_CLASSES)

    def objective_fn(config: dict) -> ObjectiveResult:
        trial_dir = _config_dir(out_dir, config)
        target_folder = trial_dir / "target"
        target_folder.mkdir(parents=True, exist_ok=True)
        epochs = int(config.get("epochs", cfg.epochs))  # epochs may itself be a knob

        # 1. Write the direct DP-SGD knobs; both target and shadow training read these.
        dpsgd_path = target_folder / "dpsgd_dic.pkl"
        with dpsgd_path.open("wb") as f:
            pickle.dump({"noise_multiplier": config["noise_multiplier"],
                         "max_grad_norm": config["max_grad_norm"],
                         "delta": cfg.delta}, f)

        # 2. Train the target under the knobs (the same dp_train the shadow models use).
        batch_size = int(config["batch_size"])
        train_loader = DataLoader(TensorDataset(x[train_idx], y[train_idx]),
                                  batch_size=batch_size, shuffle=True)
        private = config["noise_multiplier"] > 0
        model = SmallCNN(num_classes=NUM_CLASSES, dpsgd=private)
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        train_result = dp_train(model, train_loader, criterion, optimizer, epochs, str(dpsgd_path), device)
        epsilon = train_result.metrics.extra.get("epsilon")

        # 3. Utility on the disjoint test split, then the gate: a model at or
        # below chance level has not learned — attacking it spends a full audit
        # to measure nothing, and its degenerate TPR would sit on the frontier
        # looking like privacy. Prune instead.
        test_loader = DataLoader(TensorDataset(x[test_idx], y[test_idx]), batch_size=512)
        utility = evaluate_utility(train_result.model, test_loader, cfg.utility.metric, device)
        if utility <= gate:
            logger.warning(f"Utility {utility:.4f} ({cfg.utility.metric}) is at or below the gate "
                           f"{gate:.4f}: this model did not learn, skipping the attack.")
            return ObjectiveResult(utility=utility, tpr=None,
                                   extras={"epsilon": epsilon, "tuning_accounted": False, "gated": True})

        # 4. Persist the target in the layout LeakPro's MIAHandler reads.
        state_dict = {k.replace("_module.", "").replace("module.", ""): v
                      for k, v in train_result.model.state_dict().items()}
        with (target_folder / "target_model.pkl").open("wb") as f:
            torch.save(state_dict, f)
        test_result = CifarDPHandler().eval(test_loader, train_result.model, criterion)
        metadata = LeakPro.make_mia_metadata(
            train_result=train_result, optimizer=optimizer, loss_fn=criterion,
            dataloader=train_loader, test_result=test_result, epochs=epochs,
            train_indices=pop["train_indices"], test_indices=pop["test_indices"],
            dataset_name="cifar10_pet")
        with (target_folder / "model_metadata.pkl").open("wb") as f:
            pickle.dump(metadata, f)

        # 5. Audit the saved target with the real RMIA attack through the pipeline.
        audit_dir = trial_dir / "audit"
        config_path = trial_dir / "audit.yaml"
        _write_audit_yaml(config_path, pop["path"], target_folder, dpsgd_path,
                          audit_dir, cfg.rmia, cfg.seed)
        result = run_rmia_audit(CifarDPHandler, str(config_path))

        # A model that passed the gate can still saturate every RMIA score, or
        # tie so many scores that no threshold reaches the proxy FPR. Both read
        # as TPR 0 for reasons that are not privacy — an unresolved operating
        # point must prune the trial (tpr=None), exactly like the utility gate,
        # or "we could not measure it" masquerades as "perfectly private" and
        # sits on the frontier. realized_fpr/degenerate_audit stay in the
        # record so the pruning is auditable.
        tpr, realized_fpr, degenerate = resolved_proxy_tpr(result, cfg.proxy_fpr)
        if tpr is None:
            logger.warning(
                f"Audit unresolved at FPR {cfg.proxy_fpr:.2%} (realized "
                f"{'none' if realized_fpr is None else f'{realized_fpr:.4%}'}, "
                f"degenerate={degenerate}): not evidence of privacy, pruning the trial."
            )

        return ObjectiveResult(
            utility=utility, tpr=tpr,
            extras={"epsilon": epsilon,
                    # The accountant's epsilon covers ONE training run. Tuning
                    # over many configurations on the same private data and
                    # selecting off the frontier is itself a mechanism (Liu &
                    # Talwar 2019; Papernot & Steinke 2022), and that cost is
                    # not included — the flag travels with the number so a
                    # report cannot silently present it as the procedure's
                    # guarantee.
                    "tuning_accounted": False,
                    "roc_auc": result.roc_auc,
                    "realized_fpr": realized_fpr,
                    "degenerate_audit": degenerate},
        )

    return objective_fn


MIN_PROXY_EVENTS = 10


def check_proxy_resolution(cfg: PrivacyUtilityConfig) -> None:
    """Fail fast when the proxy FPR cannot be measured with this audit set.

    An FPR level outside ``fixed_fpr_table`` cannot be read back at all, and a
    level allowing fewer than ~MIN_PROXY_EVENTS false positives (e.g. 0.01% on
    2000 nonmembers = 0.2) reads 0 for every config — every trial would be
    pruned as unresolved after paying for its training and audit.
    """
    if not any(abs(cfg.proxy_fpr - f) < 1e-12 for f in TABULATED_FPRS):
        raise ValueError(
            f"proxy_fpr={cfg.proxy_fpr} is not a tabulated FPR level {TABULATED_FPRS}: "
            "TPR cannot be read from MIAResult.fixed_fpr_table at this level."
        )
    expected = cfg.splits.n_test * cfg.proxy_fpr
    if expected < MIN_PROXY_EVENTS:
        raise ValueError(
            f"proxy_fpr={cfg.proxy_fpr:.2%} on n_test={cfg.splits.n_test} nonmembers allows "
            f"~{expected:.1f} false positives — the operating point is unresolvable and every "
            f"trial would be pruned. Use n_test >= {int(MIN_PROXY_EVENTS / cfg.proxy_fpr)}, "
            "or a higher proxy_fpr."
        )


def load_settings(path: str, smoke: bool) -> PrivacyUtilityConfig:
    """Load and validate the settings yaml; --smoke shrinks it to a minutes-long pipeline check."""
    with open(path, "rb") as f:
        cfg = PrivacyUtilityConfig(**yaml.safe_load(f))
    if smoke:
        update = {
            "n_trials": 2, "epochs": 8,
            "output_dir": cfg.output_dir + "_smoke",
            "rmia": {**cfg.rmia, "num_shadow_models": 3},
            # Small but overfitting: few members trained for several epochs so
            # the target memorizes and RMIA has a real signal to find.
            # pop_size = 2 * n_target keeps shadow-size mimicry (see build_population).
            "splits": {"pop_size": 600, "n_target": 300, "n_test": 300},
            # 300 nonmembers resolve 10%, not the yaml's 1% (see check_proxy_resolution).
            "proxy_fpr": 0.1,
        }
        cfg = PrivacyUtilityConfig(**{**cfg.model_dump(), **update})
    check_proxy_resolution(cfg)
    return cfg


def run_optimization(config_path: str | None = None, smoke: bool = False, device: str | None = None):  # noqa: ANN201
    """Run the CIFAR DP-SGD privacy-utility optimization end to end; returns the Optuna study.

    This is the single entry point for both the CLI (``main``) and the
    ``cifar_dpsgd.ipynb`` notebook cell.
    """
    config_path = config_path or str(Path(__file__).resolve().parent / "dpsgd_optimization.yaml")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_settings(config_path, smoke)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pop = build_population(cfg, out_dir)
    # The resolved settings are recorded so validate_frontier audits with the same recipe.
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg.model_dump(), indent=2))

    objective_fn = make_objective(cfg, pop, out_dir, device)

    start = time.time()
    study = optimize(objective_fn, KnobSpace.from_config(cfg.knobs, cfg.fixed), out_dir,
                     n_trials=cfg.n_trials, proxy_fpr=cfg.proxy_fpr, seed=cfg.seed)
    logger.info(f"Optimization done: {len(study.trials)} trials in {time.time() - start:.0f}s.")

    front = sorted(pareto_trials(study), key=lambda t: t.values[0])
    logger.info(f"Pareto front ({len(front)} points):")
    for t in front:
        eps = t.user_attrs.get("epsilon")
        logger.info(f"  {cfg.utility.metric}={t.values[0]:.4f}  TPR@{cfg.proxy_fpr:.0%}={t.values[1]:.4f}  "
                    f"eps={'inf' if eps is None else f'{eps:.2f}'}  config={t.params}")
    plot_path = plot_frontier(study, out_dir / "frontier.png")
    logger.info(f"Frontier plot: {plot_path}")
    return study


def main() -> None:
    """CLI wrapper around :func:`run_optimization`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="settings yaml (schema: leakpro.schemas.PrivacyUtilityConfig); default: ./dpsgd_optimization.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end check (2 trials, small data)")
    args = parser.parse_args()
    run_optimization(args.config, args.smoke, args.device)


if __name__ == "__main__":
    main()
