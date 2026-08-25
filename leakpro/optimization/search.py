#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Bayesian-optimization search over PET hyperparameters.

The search explores a :class:`~leakpro.optimization.knobs.KnobSpace` for the
privacy-utility frontier of a DP-SGD target. It is *model-based*: every
configuration is proposed by an Optuna sampler from the (utility, attack-TPR)
pairs observed so far, so the search concentrates on the frontier instead of
sweeping a fixed grid.

Two objectives are optimized jointly — utility is maximized, attack TPR at a
fixed FPR (the privacy axis) is minimized — so the result is a Pareto front, read
off ``study.best_trials``.

Optuna is LeakPro's existing hyperparameter-search engine
(``leakpro/attacks/utils/hyperparameter_tuning/optuna.py`` already uses it to tune
attack parameters); reusing it here keeps the search inside the library's
toolset rather than introducing a second optimizer. Multi-objective TPE gives the
history-guided proposals; the study is persisted to a SQLite file so an
interrupted run resumes exactly where it stopped.

The search is deliberately agnostic to *how* a configuration is evaluated: the
caller passes an ``objective_fn`` that trains a target and audits it. The
canonical implementation (see ``examples/mia/cifar/dpsgd_optimization``) trains a
DP-SGD target under the knobs and audits it with
:func:`leakpro.optimization.audit.run_rmia_audit`, i.e. LeakPro's own RMIA
attack. The objective must return an :class:`ObjectiveResult` whose ``tpr`` is the
attack TPR at the search's ``proxy_fpr``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import optuna

from leakpro.optimization.knobs import KnobSpace
from leakpro.utils.logger import logger
from leakpro.utils.seed import seed_everything

DEFAULT_STUDY_NAME = "privacy_utility_optimization"


@dataclass
class ObjectiveResult:
    """The outcome of evaluating one configuration.

    Args:
        utility: Utility of the trained target (higher is better).
        tpr: Attack TPR at the run's proxy FPR (lower is safer). Must be the
            TPR at the same FPR the search was created with. ``None`` means the
            attack was skipped because the model failed the utility gate: the
            trial is pruned — it consumes budget but can never sit on the
            frontier, so a model that learned nothing cannot masquerade as
            "perfectly private".
        extras: Optional extra fields recorded on the trial (e.g. formal epsilon,
            attack ROC-AUC) for later inspection; they do not affect the search.

    """

    utility: float
    tpr: float | None
    extras: dict = field(default_factory=dict)


def optimize(  # noqa: PLR0913
    objective_fn: Callable[[dict[str, float]], ObjectiveResult],
    knob_space: KnobSpace,
    output_dir: str | Path,
    n_trials: int,
    *,
    proxy_fpr: float = 0.01,
    seed: int = 0,
    anchors: list[dict[str, float]] | None = None,
    study_name: str = DEFAULT_STUDY_NAME,
) -> optuna.study.Study:
    """Run a Bayesian-optimization search and return the Optuna study.

    Args:
        objective_fn: config -> :class:`ObjectiveResult`. Trains and audits one
            configuration. Its ``tpr`` must be the attack TPR at ``proxy_fpr``.
        knob_space: The searched configuration space.
        output_dir: Where the study database (``study.db``) is written; an
            existing database with the same ``study_name`` resumes the run.
        n_trials: Total number of configurations to evaluate over the run's
            lifetime (already-finished trials count toward this on resume).
        proxy_fpr: FPR level the optimizer minimizes TPR at (default 1%). Recorded
            on the study; the objective is responsible for reporting TPR here.
        seed: Seeds the sampler, and every RNG (torch/numpy/random) per trial
            from (seed, trial number) — so each trial's training is reproducible
            independently of how many trials ran before it in the process.
        anchors: Explicit configurations evaluated first (e.g. a non-private
            reference point that continuous search would never land on exactly).
        study_name: Name of the Optuna study inside the database.

    Returns:
        The completed :class:`optuna.study.Study`. The Pareto front is
        ``study.best_trials``.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed torch/numpy/random for any pre-loop setup (e.g. split permutation).
    # Each trial re-seeds from (seed, trial number) — see _objective.
    seed_everything(seed)

    storage = f"sqlite:///{output_dir / 'study.db'}"
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(
        directions=["maximize", "minimize"],  # utility up, attack TPR down
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    study.set_user_attr("proxy_fpr", proxy_fpr)
    study.set_user_attr("knob_space", knob_space.to_dict())
    study.set_user_attr("seed", seed)
    study.set_user_attr("output_dir", str(output_dir))

    finished = [t for t in study.trials if t.state.is_finished()]
    if finished:
        logger.info(f"Resuming: {len(finished)} finished trial(s) found in {storage}.")
    elif anchors:
        # Enqueue anchors only on a fresh study; on resume they are already run.
        for anchor in anchors:
            study.enqueue_trial(anchor)

    def _objective(trial: optuna.trial.Trial) -> tuple[float, float]:
        config = knob_space.suggest(trial)
        # Record the *resolved* configuration (searched + fixed knobs): Optuna's
        # trial.params holds only the searched ones, so anything keyed on the
        # full config (per-trial artifact dirs, re-audits) must read this attr.
        trial.set_user_attr("config", config)
        # Re-seed every RNG deterministically from (seed, trial number): the
        # global RNG state at trial N otherwise depends on how many trials ran
        # earlier in *this* process, so a resumed run would train different
        # models than a fresh one at the same trial. (The TPE sampler carries
        # its own seeded RNG and is unaffected by this.)
        seed_everything((seed * 1_000_003 + trial.number) % (2**31 - 1))
        logger.info(f"Trial {trial.number}: {config}")
        result = objective_fn(config)
        trial.set_user_attr("utility", result.utility)
        trial.set_user_attr("tpr", result.tpr)
        for key, value in result.extras.items():
            trial.set_user_attr(key, value)
        if result.tpr is None:
            logger.info(
                f"Trial {trial.number}: utility={result.utility:.4f} failed the utility gate — "
                "attack skipped, trial pruned (excluded from the frontier)."
            )
            raise optuna.TrialPruned
        logger.info(
            f"Trial {trial.number}: utility={result.utility:.4f}, "
            f"TPR@{proxy_fpr:.0%}={result.tpr:.4f}"
        )
        return result.utility, result.tpr

    remaining = max(0, n_trials - len(finished))
    if remaining == 0:
        logger.info(f"Study already has {len(finished)} finished trials (>= n_trials={n_trials}); nothing to do.")
        return study
    study.optimize(_objective, n_trials=remaining)
    return study
