#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Frontier campaign: evaluate PET configurations on (utility, attack success).

v0 is a Sobol sweep — it validates the whole pipeline (training, matched-shadow
attack, metrics, persistence) and its scatter is already a frontier estimate.
A model-based sampler (qNEHVI) replaces the Sobol draw later without touching
anything else.

The campaign is deliberately outside LeakPro's attack stack: it consumes three
callables (train, utility, attack) and imports nothing from ``leakpro`` beyond
the logger. No ``AbstractInputHandler`` is involved — the examples train and
score raw tensors directly, and the scoring path here is its own, separate from
``leakpro/signals`` and ``MIAResult``.

That separation buys a loop that can retrain and re-attack a model per sampled
configuration without carrying handler machinery, but it is a duplicate scoring
path, and keeping it means accepting that its conventions can drift from the
attack stack's. Wiring ``attack_fn`` to LeakPro's own attacks is the obvious
alternative and remains open.

Full mimicry is the adapter's contract either way: reference models inside
``attack_fn`` must be trained with the same configuration as the candidate
target.
"""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from leakpro.optimization.knobs import KnobSpace
from leakpro.optimization.objectives import AttackScores, clopper_pearson_ci, tpr_at_fpr
from leakpro.utils.logger import logger
from leakpro.utils.seed import seed_everything


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

    @property
    def _meta_path(self) -> Path:
        return self.output_dir / "campaign.json"

    def _identity(self) -> dict:
        return {"seed": self.seed, "proxy_fpr": self.proxy_fpr, "knob_space": self.knob_space.to_dict()}

    def _load_existing(self) -> None:
        # Resume is index-based, and indices only name Sobol positions of ONE
        # (seed, space) pair — mixing sweeps would silently attribute another
        # campaign's results to this one's configurations. Refuse instead.
        if self._meta_path.exists():
            stored = json.loads(self._meta_path.read_text())
            if stored != self._identity():
                raise ValueError(
                    f"{self.output_dir} holds a campaign with a different seed, proxy_fpr or knob space; "
                    "resuming would mix incompatible sweeps. Use a fresh output_dir or match the stored settings."
                )
        else:
            self._meta_path.write_text(json.dumps(self._identity(), indent=2))
        if not self._log_path.exists():
            return
        with self._log_path.open() as f:
            self.records = [EvaluationRecord(json.loads(line)) for line in f if line.strip()]
        if self.records:
            logger.info(f"Resuming campaign: {len(self.records)} evaluations found in {self._log_path}.")

    @staticmethod
    def _json_safe(value):  # noqa: ANN001, ANN205
        # float("inf") serializes as a bare `Infinity`, which is not JSON and
        # which browsers reject; the non-private anchor's epsilon is exactly
        # that. None is the documented spelling of "no finite value".
        if isinstance(value, dict):
            return {k: Campaign._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Campaign._json_safe(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _save_scores(self, index: int, scores: AttackScores) -> None:
        """Write one config's raw attack scores next to the evaluation log."""
        out = self.output_dir / "scores"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / f"config_{index}.npz",
            member_scores=scores.member_scores,
            nonmember_scores=scores.nonmember_scores,
        )

    def _append(self, record: EvaluationRecord) -> None:
        record = EvaluationRecord(self._json_safe(record))
        self.records.append(record)
        with self._log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def run(self, n_configs: int, anchors: list[dict[str, float]] | None = None) -> list[EvaluationRecord]:
        """Evaluate a Sobol sweep of ``n_configs`` configurations (resumes if interrupted).

        ``anchors`` are explicit configurations evaluated before the sweep under
        reserved negative indices (-1, -2, ...): sampling can never land on an
        exact value like ``noise_multiplier == 0``, so documented anchor points
        must be forced, not hoped for.

        A configuration that raises is recorded with an ``error`` field and the
        sweep continues; errored indices are retried on the next resume.
        """
        planned = [(-(i + 1), config) for i, config in enumerate(anchors or [])]
        planned += list(enumerate(self.knob_space.sample_sobol(n_configs, seed=self.seed)))
        done = {r["index"] for r in self.records if "error" not in r}
        for index, config in planned:
            if index in done:
                continue
            try:
                record = self._evaluate(index, config)
            except Exception as exc:  # noqa: BLE001 — one bad config must not kill or livelock the sweep
                logger.error(f"Config {index} failed: {exc}")
                record = EvaluationRecord(index=index, config=config, seed=self.seed,
                                          proxy_fpr=self.proxy_fpr, error=str(exc))
            self._append(record)
        return self.records

    def _seed_for(self, index: int) -> None:
        """Seed every RNG deterministically from (campaign seed, config index).

        Seeding once per run is not enough: resume skips finished
        configurations, so the RNG state reached at a given index depends on how
        many configurations ran before it in *this* process. A fresh run and a
        resumed run would then train different models at the same index, and
        validation would retrain a different target than the one on the
        frontier. Deriving the seed from the index makes each configuration's
        training reproducible independently of run history.
        """
        seed_everything((self.seed * 1_000_003 + index) % (2**31 - 1))

    def _evaluate(self, index: int, config: dict[str, float]) -> EvaluationRecord:
        logger.info(f"Evaluating config {index}: {config}")
        self._seed_for(index)
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
        # Persist the scores the metric was computed from. Aggregates alone made
        # the degenerate cases unauditable: establishing that a TPR of 0 came
        # from a saturated, five-distinct-value score distribution rather than
        # from real privacy required monkeypatching attack_fn.
        self._save_scores(index, scores)
        measured = tpr_at_fpr(scores, self.proxy_fpr)
        lower, upper = clopper_pearson_ci(measured.events, measured.n_members)
        record.update(
            attack_tpr=measured.tpr,
            attack_tpr_events=measured.events,
            attack_tpr_n=measured.n_members,
            attack_tpr_ci95=[lower, upper],
            # The operating point behind the number. Without these a TPR is
            # uninterpretable: an unresolvable estimate and a genuinely private
            # model both read as a bare 0.0.
            attack_realized_fpr=measured.realized_fpr,
            attack_threshold=measured.threshold,
            attack_resolution_warning=measured.warning,
            # Few distinct values means a saturated model; this is the cheap
            # in-record diagnostic for the case that made a bare TPR misleading.
            attack_distinct_nonmember_scores=int(np.unique(scores.nonmember_scores).size),
            # Binomial sampling error over audit points ONLY. It excludes
            # target-training randomness and the reference draw, which are
            # plausibly larger, so it is narrower than the true uncertainty on a
            # quantity pareto_front then minimizes over many draws.
            attack_tpr_ci95_kind="clopper_pearson_binomial_only",
        )
        if measured.warning:
            logger.warning(f"Config {index}: {measured.warning}")
        logger.info(
            f"Config {index}: utility {record['utility']:.4f}, "
            f"TPR@{self.proxy_fpr:.0%} = {measured.tpr:.4f} "
            f"({measured.events}/{measured.n_members} events, realized FPR {measured.realized_fpr:.2%})."
        )
        return record
