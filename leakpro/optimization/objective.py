#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The DP-SGD search objective: train a target under a configuration, then audit it with real RMIA.

:func:`leakpro.optimization.search.optimize` is agnostic to how a configuration is
scored; it only wants ``config -> ObjectiveResult``. This module builds that
callable for the common case, a :class:`~leakpro.optimization.training.PETRecipe`
trained with DP-SGD and attacked by :class:`~leakpro.attacks.mia_attacks.rmia.AttackRMIA`
through the ordinary :class:`~leakpro.leakpro.LeakPro` pipeline. Every trial

1. writes the trial's DP-SGD knobs to a ``dpsgd_dic.pkl`` that the target *and* the
   RMIA shadow models read, so the references are trained exactly like the candidate;
2. trains the target with :func:`~leakpro.optimization.training.fit_dpsgd`;
3. scores utility on the held-out split and prunes a model that did not learn;
4. saves the target in the layout :class:`~leakpro.input_handler.mia_handler.MIAHandler`
   expects (``target_model.pkl`` + ``model_metadata.pkl``) and a per-trial ``audit.yaml``;
5. runs :func:`~leakpro.optimization.audit.run_rmia_audit` and reports the TPR
   resolved at the proxy FPR (:func:`~leakpro.optimization.audit.resolved_proxy_tpr`).

The population file the audit reads must hold data *as the model consumes it*
(normalised, channel-first): ``MIAHandler.get_dataset`` slices ``population.data``
directly, so any preprocessing hidden in a ``__getitem__`` would reach the target
but not the shadow models. :func:`materialize_population` builds such a file from
any dataset by reading it through its own ``__getitem__`` once.

Population layout follows the mimicry rule documented in
``leakpro.optimization.audit``: the audit population is exactly twice the target's
training set, so RMIA's balanced shadow sampling trains each reference on as many
points as the target saw.
"""

import hashlib
import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.leakpro import LeakPro
from leakpro.optimization.audit import TABULATED_FPRS, resolved_proxy_tpr, run_rmia_audit, tpr_at_fixed_fpr
from leakpro.optimization.search import ObjectiveResult
from leakpro.optimization.training import (
    DEFAULT_ACCOUNTANT,
    DEFAULT_DELTA,
    DEFAULT_MAX_PHYSICAL_BATCH,
    PETRecipe,
    fit_dpsgd,
)
from leakpro.reporting.mia_result import MIAResult
from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.logger import logger
from leakpro.utils.save_load import unwrapped_state_dict

DPSGD_CONFIG_NAME = "dpsgd_dic.pkl"
POPULATION_NAME = "population.pkl"
RMIA_DEFAULTS = {"num_shadow_models": 2, "training_data_fraction": 0.5, "online": False}
# Fewer expected false positives than this at the proxy FPR and the operating
# point is unresolvable: TPR reads 0 for every configuration and every trial is
# pruned after paying for its training and audit.
MIN_PROXY_EVENTS = 10


# --------------------------------------------------------------------------- #
# Training and evaluation shared by the target and the shadow models
# --------------------------------------------------------------------------- #

def _is_binary(criterion: nn.Module) -> bool:
    return isinstance(criterion, nn.BCEWithLogitsLoss)


def _binary_loss(criterion: nn.Module) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Adapt a (B, 1) single-logit head and long labels to ``BCEWithLogitsLoss``.

    The metadata keeps ``BCEWithLogitsLoss`` itself (so the shadow models rebuild
    the same loss); only the call site reshapes. Same rule as the webapp's
    training loop.
    """
    def loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return criterion(output.view(-1), target.float().view(-1))
    return loss


def _predictions(output: torch.Tensor, binary: bool) -> torch.Tensor:
    return (output.view(-1).sigmoid() > 0.5).long() if binary else output.argmax(1)


def evaluate(dataloader: DataLoader, model: nn.Module, criterion: nn.Module,
             device: str | None = None) -> EvalOutput:
    """Accuracy and mean loss over ``dataloader``, binary-aware."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    binary = _is_binary(criterion)
    loss_fn = _binary_loss(criterion) if binary else criterion
    model = model.to(device)
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device).view(-1)
            out = model(xb)
            total_loss += loss_fn(out, yb).item() * yb.size(0)
            correct += (_predictions(out, binary) == yb.long()).sum().item()
            seen += yb.size(0)
    return EvalOutput(accuracy=correct / max(seen, 1), loss=total_loss / max(seen, 1))


def load_dpsgd_config(dpsgd_path: str | Path) -> dict:
    """Read the DP-SGD knobs a trial wrote for its target and shadow models."""
    path = Path(dpsgd_path)
    if not path.exists():
        raise FileNotFoundError(f"DP-SGD config not found: {dpsgd_path}")
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301 - written by this module, in the trial directory


def dp_train(  # noqa: PLR0913
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    dpsgd_path: str | Path,
    device: str | None = None,
) -> TrainingOutput:
    """Train one model with the knobs in ``dpsgd_path``; the one path for target and shadow models.

    ``noise_multiplier == 0`` is the non-private anchor and runs the plain loop.
    The accountant and the physical batch cap travel in the same file so a shadow
    model is trained under exactly the target's regime.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_dpsgd_config(dpsgd_path)
    loss_fn = _binary_loss(criterion) if _is_binary(criterion) else criterion
    model = fit_dpsgd(
        model, dataloader, loss_fn, optimizer, epochs,
        noise_multiplier=float(cfg["noise_multiplier"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        device=device,
        delta=float(cfg.get("delta", DEFAULT_DELTA)),
        max_physical_batch=int(cfg.get("max_physical_batch", DEFAULT_MAX_PHYSICAL_BATCH)),
        accountant=str(cfg.get("accountant", DEFAULT_ACCOUNTANT)),
    )
    # Training-set fit after the last epoch, on the logical loader (the Poisson-
    # sampled private loader does not visit every example exactly once).
    fit = evaluate(dataloader, model, criterion, device)
    model.to("cpu")
    return TrainingOutput(model=model, metrics=EvalOutput(accuracy=fit.accuracy, loss=fit.loss,
                                                          extra=dict(model.dp_accounting)))


class DPSGDAuditHandler(AbstractInputHandler, role="full"):
    """Input handler for auditing a DP-SGD target: shadow models train with the target's own knobs.

    ``train`` and ``eval`` are rebound onto the :class:`MIAHandler` by LeakPro, so
    ``self.configs`` is the audit config and ``configs.target.dpsgd_path`` points
    at the trial's ``dpsgd_dic.pkl``.
    """

    class UserDataset(AbstractInputHandler.UserDataset):
        """Pre-normalised tensors; ``data[i]`` is exactly what the model consumes."""

        def __init__(self, data: torch.Tensor, targets: torch.Tensor, **kwargs: dict) -> None:
            self.data = data
            self.targets = targets
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __len__(self) -> int:
            """Number of samples."""
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple:
            """One ``(x, y)`` pair, as stored."""
            return self.data[index], self.targets[index]

    def train(  # noqa: PLR0913
        self,
        dataloader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
    ) -> TrainingOutput:
        """Train a shadow (or target) model under the trial's DP-SGD knobs."""
        dpsgd_path = getattr(self.configs.target, "dpsgd_path", None)
        if not dpsgd_path:
            raise ValueError("target.dpsgd_path must be set: the DP-SGD audit handler trains from it.")
        return dp_train(model, dataloader, criterion, optimizer, epochs, dpsgd_path)

    def eval(self, dataloader: DataLoader, model: nn.Module, criterion: nn.Module) -> EvalOutput:
        """Accuracy and mean loss over ``dataloader``."""
        return evaluate(dataloader, model, criterion)


# --------------------------------------------------------------------------- #
# Population
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Population:
    """An audit population on disk plus the index roles the objective uses.

    ``train_indices`` are the target's members; ``test_indices`` are the
    non-members and the utility split. Both index into the population file.
    """

    path: str
    train_indices: list[int]
    test_indices: list[int]
    num_classes: int


def materialize_population(  # noqa: PLR0913
    dataset: Dataset,
    out_dir: str | Path,
    *,
    n_target: int,
    n_test: int | None = None,
    seed: int = 0,
    name: str = POPULATION_NAME,
) -> Population:
    """Write a normalised audit population of ``2 * n_target`` samples and carve its roles.

    Samples are read through ``dataset.__getitem__`` so whatever normalisation the
    dataset applies is baked into the file; the audit then slices tensors.

    Args:
        dataset: the full dataset, ``__getitem__ -> (x, y)``.
        out_dir: where the population file is written.
        n_target: the target's training-set size; the population is twice this
            (the RMIA mimicry rule), so ``2 * n_target <= len(dataset)``.
        n_test: held-out size (non-members and utility), at most ``n_target``;
            defaults to ``n_target``.
        seed: permutation seed.
        name: population file name.

    """
    n = len(dataset)
    if 2 * n_target > n:
        raise ValueError(f"n_target={n_target} needs a dataset of at least {2 * n_target} samples, got {n}.")
    n_test = n_target if n_test is None else n_test
    if not 0 < n_test <= n_target:
        raise ValueError(f"n_test={n_test} must be in (0, n_target={n_target}].")

    order = np.random.default_rng(seed).permutation(n)[:2 * n_target]
    loader = DataLoader(Subset(dataset, [int(i) for i in order]), batch_size=512, shuffle=False)
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb.float())
        ys.append(yb.view(-1).long())
    data, targets = torch.cat(xs), torch.cat(ys)
    population = DPSGDAuditHandler.UserDataset(data, targets)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    with path.open("wb") as f:
        pickle.dump(population, f)
    logger.info(f"Population of {len(population)} samples written to {path} "
                f"(members {n_target}, held-out {n_test}).")
    return Population(path=str(path), train_indices=list(range(n_target)),
                      test_indices=list(range(n_target, n_target + n_test)),
                      num_classes=int(targets.max().item()) + 1)


def resolution_warning(n_nonmembers: int, proxy_fpr: float) -> str | None:
    """Say why ``proxy_fpr`` cannot be measured on ``n_nonmembers`` points, or None when it can.

    An FPR outside ``fixed_fpr_table`` cannot be read back at all; one allowing
    fewer than ``MIN_PROXY_EVENTS`` false positives reads 0 for every configuration.
    """
    if not any(abs(proxy_fpr - f) < 1e-12 for f in TABULATED_FPRS):
        return (f"proxy_fpr={proxy_fpr} is not a tabulated FPR level {TABULATED_FPRS}; "
                "TPR cannot be read from MIAResult.fixed_fpr_table at this level.")
    expected = n_nonmembers * proxy_fpr
    if expected < MIN_PROXY_EVENTS:
        return (f"proxy_fpr={proxy_fpr:.2%} on {n_nonmembers} non-members allows ~{expected:.1f} false "
                f"positives; fewer than {MIN_PROXY_EVENTS} makes the operating point unresolvable and "
                f"every trial would be pruned. Use at least {int(MIN_PROXY_EVENTS / proxy_fpr)} "
                "non-members, or a higher proxy FPR.")
    return None


# --------------------------------------------------------------------------- #
# Per-trial artifacts
# --------------------------------------------------------------------------- #

def config_dir(out_dir: str | Path, config: dict) -> Path:
    """Stable per-configuration directory, so a resumed run reuses a config's artifacts."""
    key = json.dumps({k: round(float(v), 8) for k, v in sorted(config.items())})
    tag = hashlib.sha1(key.encode()).hexdigest()[:10]  # noqa: S324 - a directory name, not security
    return Path(out_dir) / "trials" / f"config_{tag}"


def write_audit_yaml(  # noqa: PLR0913
    path: Path,
    *,
    population_path: str,
    target_folder: Path,
    dpsgd_path: Path,
    audit_dir: Path,
    module_path: str,
    model_class: str,
    rmia_config: dict,
    seed: int,
    data_modality: str = "image",
    shadow_init_params: dict | None = None,
) -> None:
    """Write a per-trial audit config that runs RMIA against the saved target."""
    config = {
        "audit": {
            "random_seed": seed,
            "attack_list": [{"attack": "rmia", **rmia_config}],
            "output_dir": str(audit_dir),
            "attack_type": "mia",
            "data_modality": data_modality,
        },
        "target": {
            "module_path": str(module_path),
            "model_class": model_class,
            "target_folder": str(target_folder),
            "data_path": population_path,
            "dpsgd_path": str(dpsgd_path),
        },
    }
    if shadow_init_params:
        config["shadow_model"] = {"init_params": dict(shadow_init_params)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f)


@dataclass(frozen=True)
class AuditTarget:
    """What the objective needs to know to audit a trained model.

    ``init_params`` are the constructor kwargs the target was built with. LeakPro
    records only the kwargs an architecture stores as attributes
    (``get_model_init_params``); passing them here writes them into the audit
    config's ``shadow_model.init_params`` as well, so the shadow models are built
    like the target even when the architecture does not echo its arguments.
    """

    module_path: str
    model_class: str
    dataset_name: str = "population"
    data_modality: str = "image"
    init_params: dict | None = None


def make_dpsgd_objective(  # noqa: PLR0913
    recipe: PETRecipe,
    population: Population,
    target: AuditTarget,
    out_dir: str | Path,
    *,
    proxy_fpr: float = 0.01,
    delta: float = DEFAULT_DELTA,
    accountant: str = DEFAULT_ACCOUNTANT,
    max_physical_batch: int = DEFAULT_MAX_PHYSICAL_BATCH,
    rmia_config: dict | None = None,
    seed: int = 0,
    device: str | None = None,
    utility_gate: float | None = None,
    handler_cls: type[AbstractInputHandler] = DPSGDAuditHandler,
) -> Callable[[dict], ObjectiveResult]:
    """Build ``config -> ObjectiveResult`` for :func:`~leakpro.optimization.search.optimize`.

    Args:
        recipe: model / optimizer / loader factories and the loss.
        population: the audit population and its index roles (:func:`materialize_population`).
        target: how the audit should rebuild the model (arch file and class name).
        out_dir: run directory; trials live under ``out_dir/trials``.
        proxy_fpr: the FPR the search minimises TPR at; must be tabulated.
        delta: DP delta for epsilon accounting.
        accountant: ``"prv"`` or ``"rdp"``; recorded next to every epsilon.
        max_physical_batch: per-sample-gradient memory cap, speed only.
        rmia_config: RMIA parameters for the per-trial audit; defaults to ``RMIA_DEFAULTS``.
        seed: audit seed.
        device: training device; auto-detected when None.
        utility_gate: utility at or below which the model "did not learn" and the
            trial is pruned without an audit; defaults to chance level.
        handler_cls: the input handler LeakPro audits with.

    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rmia_config = {**RMIA_DEFAULTS, **(rmia_config or {})}
    binary = _is_binary(recipe.criterion)
    gate = utility_gate if utility_gate is not None else (0.5 if binary else 1.0 / population.num_classes)

    with open(population.path, "rb") as f:
        pop = pickle.load(f)  # noqa: S301 - written by materialize_population
    train_idx = np.asarray(population.train_indices)
    test_idx = np.asarray(population.test_indices)
    test_loader = DataLoader(Subset(pop, [int(i) for i in test_idx]), batch_size=512)

    def objective_fn(config: dict) -> ObjectiveResult:
        trial_dir = config_dir(out_dir, config)
        target_folder = trial_dir / "target"
        target_folder.mkdir(parents=True, exist_ok=True)
        epochs = int(config.get("epochs", recipe.epochs))

        # 1. The DP-SGD knobs both the target and the shadow models read.
        dpsgd_path = target_folder / DPSGD_CONFIG_NAME
        with dpsgd_path.open("wb") as f:
            pickle.dump({"noise_multiplier": float(config["noise_multiplier"]),
                         "max_grad_norm": float(config["max_grad_norm"]),
                         "delta": delta, "accountant": accountant,
                         "max_physical_batch": max_physical_batch}, f)

        # 2. Train the target through the same path the shadow models use.
        train_loader = recipe.make_loader(train_idx, config)
        model = recipe.make_model(config)
        optimizer = recipe.make_optimizer(model.parameters(), config)
        train_result = dp_train(model, train_loader, recipe.criterion, optimizer, epochs, dpsgd_path, device)
        epsilon = train_result.metrics.extra.get("epsilon")
        acc_name = train_result.metrics.extra.get("accountant")

        # 3. Utility, then the gate: a model at chance level has not learned, and
        # its degenerate TPR would sit on the frontier looking like privacy.
        test_result = evaluate(test_loader, train_result.model, recipe.criterion, device)
        utility = test_result.accuracy
        if utility <= gate:
            logger.warning(f"Utility {utility:.4f} is at or below the gate {gate:.4f}: "
                           "this model did not learn, skipping the attack.")
            return ObjectiveResult(utility=utility, tpr=None,
                                   extras={"epsilon": epsilon, "accountant": acc_name, "delta": delta,
                                           "tuning_accounted": False, "gated": True})

        # 4. Persist the target in the layout MIAHandler reads.
        with (target_folder / "target_model.pkl").open("wb") as f:
            torch.save(unwrapped_state_dict(train_result.model), f)
        metadata = LeakPro.make_mia_metadata(
            train_result=train_result, optimizer=optimizer, loss_fn=recipe.criterion,
            dataloader=train_loader, test_result=test_result, epochs=epochs,
            train_indices=population.train_indices, test_indices=population.test_indices,
            dataset_name=target.dataset_name)
        with (target_folder / "model_metadata.pkl").open("wb") as f:
            pickle.dump(metadata, f)

        # 5. Audit the saved target with the real RMIA attack through the pipeline.
        config_path = trial_dir / "audit.yaml"
        write_audit_yaml(config_path, population_path=population.path, target_folder=target_folder,
                         dpsgd_path=dpsgd_path, audit_dir=trial_dir / "audit",
                         module_path=target.module_path, model_class=target.model_class,
                         rmia_config=rmia_config, seed=seed, data_modality=target.data_modality,
                         shadow_init_params=target.init_params)
        result = run_rmia_audit(handler_cls, str(config_path))

        tpr, realized_fpr, degenerate = resolved_proxy_tpr(result, proxy_fpr)
        if tpr is None:
            logger.warning("Degenerate audit (no ROC; every score saturated): not evidence of privacy, "
                           "pruning the trial.")
        return ObjectiveResult(
            utility=utility, tpr=tpr,
            extras={"epsilon": epsilon, "accountant": acc_name, "delta": delta,
                    "tpr_fixed_fpr_table": None if degenerate else tpr_at_fixed_fpr(result, proxy_fpr),
                    # The accountant's epsilon covers ONE training run; selecting
                    # off a frontier of many is itself a mechanism whose cost is
                    # not included (Liu & Talwar 2019; Papernot & Steinke 2022).
                    "tuning_accounted": False,
                    "roc_auc": result.roc_auc,
                    "realized_fpr": realized_fpr,
                    "degenerate_audit": degenerate,
                    "trial_dir": str(trial_dir)},
        )

    return objective_fn


def reaudit_trial(  # noqa: PLR0913
    trial_dir: str | Path,
    *,
    population_path: str,
    target: AuditTarget,
    rmia_config: dict,
    seed: int = 0,
    tag: str = "verification",
    handler_cls: type[AbstractInputHandler] = DPSGDAuditHandler,
) -> MIAResult:
    """Re-run RMIA on a trial's saved target, e.g. with more shadow models than the search used.

    The target is not retrained: the number the user adopts is measured on the
    exact model the search produced. A fresh audit directory means fresh shadow
    models rather than the search's cached ones.
    """
    trial_dir = Path(trial_dir)
    target_folder = trial_dir / "target"
    if not (target_folder / "target_model.pkl").exists():
        raise FileNotFoundError(f"{trial_dir} holds no saved target (the trial was pruned or never ran).")
    config_path = trial_dir / f"{tag}_audit.yaml"
    write_audit_yaml(config_path, population_path=population_path, target_folder=target_folder,
                     dpsgd_path=target_folder / DPSGD_CONFIG_NAME, audit_dir=trial_dir / tag,
                     module_path=target.module_path, model_class=target.model_class,
                     rmia_config={**RMIA_DEFAULTS, **rmia_config}, seed=seed,
                     data_modality=target.data_modality, shadow_init_params=target.init_params)
    return run_rmia_audit(handler_cls, str(config_path))
