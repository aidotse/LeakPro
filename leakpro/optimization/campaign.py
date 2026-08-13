#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Frontier campaign: evaluate PET configurations on (utility, attack success).

v0 is a Sobol sweep — it validates the whole pipeline (training, matched-shadow
attack, metrics, persistence) and its scatter is already a frontier estimate.
A model-based sampler (qNEHVI) replaces the Sobol draw later without touching
anything else.

The campaign is decoupled from LeakPro internals on purpose: it consumes three
callables (train, utility, attack), and the example-side adapter maps sampled
knob values into an ``AbstractInputHandler`` recipe. Full mimicry is the
adapter's contract: reference models inside ``attack_fn`` must be trained with
the same configuration as the candidate target.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from leakpro.optimization.knobs import KnobSpace
from leakpro.optimization.objectives import AttackScores, clopper_pearson_ci, tpr_at_fpr
from leakpro.utils.logger import logger


class EvaluationRecord(dict):
    """One evaluated configuration; a dict with attribute access for the common keys."""

    @property
    def config(self) -> dict[str, float]:
        """The evaluated knob configuration."""
        return self["config"]

    @property
    def utility(self) -> float:
        """Utility of the trained model (higher is better)."""
        return self["utility"]

    @property
    def attack_tpr(self) -> float | None:
        """Attack TPR at the proxy FPR, or None if the attack was skipped."""
        return self.get("attack_tpr")


class Campaign:
    """Sobol-sweep frontier campaign over a knob space.

    Args:
        train_fn: config -> trained model (opaque to the campaign).
        utility_fn: model -> float, higher is better.
        attack_fn: (model, config) -> AttackScores from a fully-mimicked attack.
        knob_space: the searched configuration space.
        output_dir: evaluations are appended to ``evaluations.jsonl`` here;
            an existing file resumes the campaign (completed indices are skipped).
        proxy_fpr: FPR level of the optimization proxy (default 1% — the loop
            never optimizes tail statistics).
        utility_gate: optional (utility, history) -> bool; the attack runs only
            when the gate passes. Default: always attack.
        seed: Sobol scrambling seed; fixed seed + fixed space = same sweep.

    """

    def __init__(  # noqa: PLR0913
        self,
        train_fn: Callable[[dict[str, float]], Any],
        utility_fn: Callable[[Any], float],
        attack_fn: Callable[[Any, dict[str, float]], AttackScores],
        knob_space: KnobSpace,
        output_dir: str | Path,
        proxy_fpr: float = 0.01,
        utility_gate: Callable[[float, list[EvaluationRecord]], bool] | None = None,
        seed: int = 0,
    ) -> None:
        self.train_fn = train_fn
        self.utility_fn = utility_fn
        self.attack_fn = attack_fn
        self.knob_space = knob_space
        self.output_dir = Path(output_dir)
        self.proxy_fpr = proxy_fpr
        self.utility_gate = utility_gate
        self.seed = seed
        self.records: list[EvaluationRecord] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    @property
    def _log_path(self) -> Path:
        return self.output_dir / "evaluations.jsonl"

    def _load_existing(self) -> None:
        if not self._log_path.exists():
            return
        with self._log_path.open() as f:
            self.records = [EvaluationRecord(json.loads(line)) for line in f if line.strip()]
        if self.records:
            logger.info(f"Resuming campaign: {len(self.records)} evaluations found in {self._log_path}.")

    def _append(self, record: EvaluationRecord) -> None:
        self.records.append(record)
        with self._log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def run(self, n_configs: int) -> list[EvaluationRecord]:
        """Evaluate a Sobol sweep of ``n_configs`` configurations (resumes if interrupted)."""
        configs = self.knob_space.sample_sobol(n_configs, seed=self.seed)
        done = {r["index"] for r in self.records}
        for index, config in enumerate(configs):
            if index in done:
                continue
            self._append(self._evaluate(index, config))
        return self.records

    def _evaluate(self, index: int, config: dict[str, float]) -> EvaluationRecord:
        logger.info(f"Evaluating config {index}: {config}")
        record = EvaluationRecord(index=index, config=config, seed=self.seed, proxy_fpr=self.proxy_fpr)
        model = self.train_fn(config)
        extras = getattr(model, "campaign_extras", None)
        if extras:
            record.update(extras)  # e.g. formal epsilon, train/test gap
        record["utility"] = float(self.utility_fn(model))

        if self.utility_gate is not None and not self.utility_gate(record["utility"], self.records):
            record["attack_skipped"] = "utility_gate"
            logger.info(f"Config {index}: utility {record['utility']:.4f} below gate, attack skipped.")
            return record

        scores = self.attack_fn(model, config)
        tpr, k, n = tpr_at_fpr(scores, self.proxy_fpr)
        lower, upper = clopper_pearson_ci(k, n)
        record.update(
            attack_tpr=tpr,
            attack_tpr_events=k,
            attack_tpr_n=n,
            attack_tpr_ci95=[lower, upper],
        )
        logger.info(
            f"Config {index}: utility {record['utility']:.4f}, "
            f"TPR@{self.proxy_fpr:.0%} = {tpr:.4f} ({k}/{n} events)."
        )
        return record
