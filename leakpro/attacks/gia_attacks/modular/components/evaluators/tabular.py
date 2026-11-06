#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tabular reconstruction quality: tolerance matching, calibrated against a prior.

What makes a tabular reconstruction "good"?
-------------------------------------------
Unlike an image, you cannot eyeball a reconstructed table, so the evaluation has
to answer three separate questions that are easy to conflate:

1. **Is a single value right?**  Categorical values must match exactly;
   continuous values count as recovered if they land within ``tol`` standard
   deviations of the truth (``tol=0.319`` follows TabLeak, chosen so a Gaussian
   feature has a 25% chance of a hit under a random guess).

2. **Is a whole row the right row?**  A reconstructed batch is a *set* — the
   attack never learns the victim's row order — so rows must be optimally
   matched to the ground truth before any per-value comparison means anything.
   Scoring row *i* against row *i* silently reports near-chance accuracy for a
   perfect but permuted reconstruction.

3. **Is the resulting number actually good?**  This is the question accuracy
   alone does not answer.  A 70% match rate sounds strong until you notice that guessing every
   column from its marginal distribution — which requires no gradient leakage at all —
   may score 65% on the same table.  When ``reference_rows`` (data from the same
   distribution, not the attacked batch) are given, ``reconstruction_accuracy`` is
   reported alongside ``baseline_accuracy`` (the same metric for rows drawn column by
   column from the reference marginals) and ``relative_gain``:

       relative_gain = (accuracy - baseline) / (100 - baseline)   (both in percent)

   which is 0 for an attack that learned nothing beyond the marginals and 1 for
   a perfect reconstruction.  Without ``reference_rows`` both are omitted: the
   evaluator never builds a baseline from the attacked batch itself.

Row-count mismatch is reported, never hidden
--------------------------------------------
When the attack recovers a different number of rows than the victim held, the
assignment is solved rectangularly (scipy handles non-square cost matrices), no
row is discarded, and ``row_precision`` / ``row_recall`` / ``n_true`` /
``n_recon`` make the mismatch visible.

Scales: ``reconstruction_accuracy`` and ``baseline_accuracy`` are percentages (0-100);
every other accuracy (``continuous_accuracy``, ``prior_accuracy``, ...) is a fraction (0-1).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.tabular.schema import TabularSchema
from leakpro.attacks.gia_attacks.modular.components.utils.matching import hungarian_match
from leakpro.attacks.gia_attacks.modular.core.evaluation import EvalResult
from leakpro.attacks.gia_attacks.modular.core.interfaces import Evaluator

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State

# Fraction of a feature's standard deviation within which a continuous value
# counts as recovered (TabLeak, Vero et al., ICML 2023).
DEFAULT_TOLERANCE = 0.319


StdSource = Literal["batch", "unit", "reference"]
Matching = Literal["hit", "l1"]


def _hit_matrix(
    true_rows: Tensor,
    recon_rows: Tensor,
    tolerance: Tensor,
    categorical_mask: Tensor,
) -> Tensor:
    """Boolean ``[n_true, n_recon, F]`` tensor marking whether column f of each (true, recon) pair is a hit.

    Categorical columns require exact equality; continuous columns require
    ``|a - b| <= tolerance[f]``.  Computed for every (true, recon) pair at once
    because both the assignment cost and the per-column metrics need it.
    """
    diff = (true_rows.unsqueeze(1) - recon_rows.unsqueeze(0)).abs()  # [T, R, F]
    exact = diff == 0
    within = diff <= tolerance.view(1, 1, -1)
    return torch.where(categorical_mask.view(1, 1, -1), exact, within)


class TabularEvaluator(Evaluator):
    """Scores mixed continuous/categorical rows against ground truth.

    Handles both column encodings the framework produces — one-hot blocks from
    ``TabularRepresentation`` and scalar codes from column-space attacks
    — by collapsing each to the schema's canonical scalar space first.  This is
    the single tabular evaluator: row matching, thresholdable metrics and a baseline.

    Args:
        schema: Column layout; also supplies display names for per-column metrics.
        tol: Continuous tolerance as a multiple of each feature's own std.
        baseline_draws: Number of marginal-only reconstructions (drawn from ``reference_rows``) to
            average when computing ``baseline_accuracy``.  More draws means a less noisy
            calibration point; 3 is plenty for the batch sizes GIA works at.
        baseline_seed: Seed for the *local* RNG used to draw baselines.  A
            ``numpy.random.Generator`` is used rather than ``np.random.seed``
            so evaluation never perturbs global RNG state.
        report_per_column: Emit ``accuracy/<feature>`` for each column — the
            aggregate hides that an attack may fully recover a sensitive column
            while failing on filler ones.

    """

    def __init__(
        self,
        schema: TabularSchema,
        *,
        tol: float = DEFAULT_TOLERANCE,
        baseline_draws: int = 3,
        baseline_seed: int = 0,
        report_per_column: bool = True,
        std_source: StdSource = "batch",
        matching: Matching = "hit",
        reference_rows: Tensor | None = None,
        random_trials: int = 0,
        random_seed: int = 0,
        emr_thresholds: tuple[float, ...] = (0.9, 0.8, 0.6),
        round_categorical: bool = False,
    ) -> None:
        """Configure tolerance, matching and baseline options.

        Options:

        std_source: Where the continuous tolerance's std comes from.
            ``"batch"`` (default) measures it on the ground-truth batch;
            ``"unit"`` uses 1.0 for every continuous column, i.e. the tolerance is
            ``tol`` in standardised space (equals ``tol * client_std`` in raw units
            when the data was standardised with the client's own mean/std); ``"reference"``
            measures it on ``reference_rows``.
        matching: Row-assignment cost. ``"hit"`` (default) minimises 1 - hit-rate;
            ``"l1"`` minimises the L1 distance between encoded rows (categorical
            mismatch costs 2, as for one-hot vectors).
        reference_rows: Rows from the same distribution as the attacked batch (e.g. the client's
            whole dataset), in the same encoding. Enables ``baseline_accuracy`` / ``relative_gain``,
            the prior baseline (column mean / marginal mode) and, with ``random_trials > 0``, the
            random baseline.
        random_trials: Monte-Carlo trials for the random baseline.
        emr_thresholds: Row-accuracy thresholds for ``emr_<pct>`` metrics.
        round_categorical: Round scalar categorical codes before comparing.
        """
        if std_source not in ("batch", "unit", "reference"):
            raise ValueError("std_source must be 'batch', 'unit' or 'reference'.")
        if matching not in ("hit", "l1"):
            raise ValueError("matching must be 'hit' or 'l1'.")
        if std_source == "reference" and reference_rows is None:
            raise ValueError("std_source='reference' requires reference_rows.")
        self.std_source = std_source
        self.matching = matching
        self.reference_rows = reference_rows
        self.random_trials = random_trials
        self.random_seed = random_seed
        self.emr_thresholds = emr_thresholds
        self.round_categorical = round_categorical
        self.schema = schema
        self.tol = tol
        self.baseline_draws = baseline_draws
        self.baseline_seed = baseline_seed
        self.report_per_column = report_per_column

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        recon: Tensor,
        target: Tensor,
        *,
        ctx: Context | None = None,  # noqa: ARG002 -- interface signature
        state: State | None = None,  # noqa: ARG002 -- interface signature
    ) -> EvalResult:
        """Matches reconstructed rows to true rows and reports per-column hit and exact-match metrics."""
        true_rows = self.schema.to_scalar_codes(target.detach().cpu()).float()
        recon_rows = self.schema.to_scalar_codes(recon.detach().cpu()).float()

        if self.round_categorical:
            recon_rows[:, self.schema.scalar_categorical_idx] = recon_rows[
                :, self.schema.scalar_categorical_idx
            ].round()

        tolerance, cat_mask = self._tolerance_and_mask(true_rows)
        scored = self._score(true_rows, recon_rows, tolerance, cat_mask)

        metrics = dict(scored.metrics)
        if self.reference_rows is not None:
            metrics.update(self._prior_and_random(metrics, true_rows, tolerance, cat_mask))
            baseline = self._baseline_accuracy(true_rows, tolerance, cat_mask)
            metrics["baseline_accuracy"] = baseline
            metrics["relative_gain"] = _relative_gain(metrics["reconstruction_accuracy"], baseline)
        metrics["nn_reidentification_rate"] = self._reidentification_rate(
            true_rows, recon_rows, tolerance, cat_mask
        )

        return EvalResult(metrics=metrics, extras=scored.extras)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _tolerance_and_mask(self, true_rows: Tensor) -> tuple[Tensor, Tensor]:
        """Per-column tolerance band and categorical flag, in canonical order.

        The band is measured from the *ground truth* batch, so it is the spread
        of the data the attacker was trying to recover.
        """
        if self.std_source == "unit":
            stds = torch.ones(true_rows.shape[-1])
        elif self.std_source == "reference":
            stds = self.schema.scalar_col_stds(self._reference_scalar())
        else:
            stds = self.schema.scalar_col_stds(true_rows)
        tolerance = self.tol * stds
        cat_mask = torch.zeros(true_rows.shape[-1], dtype=torch.bool)
        for i in self.schema.scalar_categorical_idx:
            cat_mask[i] = True
        # Categorical tolerance is unused (exact match) but must not be NaN.
        tolerance = torch.where(cat_mask, torch.zeros_like(tolerance), tolerance)
        return tolerance, cat_mask

    def _score(
        self,
        true_rows: Tensor,
        recon_rows: Tensor,
        tolerance: Tensor,
        cat_mask: Tensor,
    ) -> EvalResult:
        n_true, n_recon = true_rows.shape[0], recon_rows.shape[0]
        n_features = true_rows.shape[-1]

        hits = _hit_matrix(true_rows, recon_rows, tolerance, cat_mask)  # [T, R, F]
        pair_accuracy = hits.float().mean(dim=-1)                       # [T, R]

        # Rectangular assignment: every row on the shorter side is matched, and
        # nothing is discarded at random to force a square problem.
        if self.matching == "l1":
            diff = (true_rows.unsqueeze(1) - recon_rows.unsqueeze(0)).abs()
            cost = torch.where(cat_mask.view(1, 1, -1), 2.0 * (diff != 0).float(), diff).sum(-1)
        else:
            cost = 1.0 - pair_accuracy
        row_ind, col_ind = hungarian_match(cost)

        matched_hits = hits[row_ind, col_ind]                # [M, F]
        matched_accuracy = matched_hits.float().mean().item()
        per_row_exact = matched_hits.all(dim=-1)             # [M]
        n_exact = int(per_row_exact.sum().item())

        metrics: dict[str, float] = {
            "reconstruction_accuracy": matched_accuracy * 100.0,
            "n_true": float(n_true),
            "n_recon": float(n_recon),
            "n_matched": float(len(row_ind)),
            # Exact whole-row recovery — the strongest privacy statement, and
            # the one that does not benefit from partially-right columns.
            "exact_row_recovery": n_exact / max(n_true, 1),
            "row_recall": n_exact / max(n_true, 1),
            "row_precision": n_exact / max(n_recon, 1),
        }
        metrics["row_f1"] = _f1(metrics["row_precision"], metrics["row_recall"])
        row_acc = matched_hits.float().mean(dim=-1)
        for t in self.emr_thresholds:
            metrics[f"emr_{int(round(t * 100))}"] = float((row_acc >= t - 1e-9).sum().item()) / max(n_true, 1)

        cont_idx = [i for i in range(n_features) if not bool(cat_mask[i])]
        cat_idx = self.schema.scalar_categorical_idx
        if cont_idx:
            metrics["continuous_accuracy"] = matched_hits[:, cont_idx].float().mean().item()
            # Error in units of each feature's std, so it is comparable across
            # columns and across datasets (raw MSE is not).
            # tolerance == tol * std for the configured std_source (batch by default).
            stds = (tolerance / self.tol if self.tol > 0 else self.schema.scalar_col_stds(true_rows)).clamp_min(1e-12)
            err = (true_rows[row_ind][:, cont_idx] - recon_rows[col_ind][:, cont_idx]).abs()
            err_sigma = err / stds[cont_idx].view(1, -1)
            metrics["mae_sigma"] = err_sigma.mean().item()
            metrics["rmse_sigma"] = err_sigma.pow(2).mean().sqrt().item()
        if cat_idx:
            metrics["categorical_accuracy"] = matched_hits[:, cat_idx].float().mean().item()

        if self.report_per_column:
            names = self.schema.scalar_feature_names
            per_col = matched_hits.float().mean(dim=0)
            for i, name in enumerate(names):
                metrics[f"accuracy/{name}"] = per_col[i].item()

        return EvalResult(
            metrics=metrics,
            extras={
                "permutation": col_ind.tolist(),
                "matched_recon": recon_rows[col_ind],
                "matched_true": true_rows[row_ind],
                "per_row_exact": per_row_exact,
            },
        )

    def _reference_scalar(self) -> Tensor:
        return self.schema.to_scalar_codes(self.reference_rows.detach().cpu()).float()

    def _prior_and_random(  # noqa: C901 -- one sequential routine
        self, metrics: dict[str, float], true_rows: Tensor, tolerance: Tensor, cat_mask: Tensor
    ) -> dict[str, float]:
        """Prior (column mean / marginal mode of ``reference_rows``) and uniform random reconstruction baselines.

        Accuracies are on a 0-1 scale (like ``continuous_accuracy``); ``gain_*`` are
        absolute differences, not headroom-normalised like ``relative_gain``.
        """
        ref = self._reference_scalar()
        n_true, n_feat = true_rows.shape
        cat_idx = self.schema.scalar_categorical_idx
        cont_idx = [i for i in range(n_feat) if not bool(cat_mask[i])]
        out: dict[str, float] = {}

        def record(prefix: str, scored: EvalResult) -> None:
            m = scored.metrics
            out[f"{prefix}_accuracy"] = m["reconstruction_accuracy"] / 100.0
            if "continuous_accuracy" in m:
                out[f"{prefix}_continuous_accuracy"] = m["continuous_accuracy"]
            if "categorical_accuracy" in m:
                out[f"{prefix}_categorical_accuracy"] = m["categorical_accuracy"]

        def gains(prefix: str) -> None:
            out[f"gain_over_{prefix}"] = metrics["reconstruction_accuracy"] / 100.0 - out[f"{prefix}_accuracy"]
            for key in ("continuous", "categorical"):
                if f"{key}_accuracy" in metrics and f"{prefix}_{key}_accuracy" in out:
                    out[f"gain_{key}_over_{prefix}"] = metrics[f"{key}_accuracy"] - out[f"{prefix}_{key}_accuracy"]

        # Prior: every row is the client mean (continuous) / marginal mode (categorical).
        prior_row = ref.mean(dim=0)
        for i in cat_idx:
            prior_row[i] = torch.mode(ref[:, i]).values
        prior = self._score(true_rows, prior_row.unsqueeze(0).repeat(n_true, 1), tolerance, cat_mask)
        record("prior", prior)
        gains("prior")

        if self.random_trials > 0:
            rng = np.random.default_rng(self.random_seed)
            lo, hi = ref.min(dim=0).values.numpy(), ref.max(dim=0).values.numpy()
            n_cats = {i: int(ref[:, i].max().item()) + 1 for i in cat_idx}
            for g, i in enumerate(cat_idx):
                if self.schema.is_one_hot:
                    n_cats[i] = self.schema.categorical_col_groups[g][1]
            accs: dict[str, list[float]] = {"reconstruction_accuracy": [], "continuous_accuracy": [], "categorical_accuracy": []}
            for _ in range(self.random_trials):
                draw = np.zeros((n_true, n_feat), dtype=np.float32)
                for f in cont_idx:
                    draw[:, f] = rng.uniform(lo[f], hi[f], size=n_true)
                for f in cat_idx:
                    draw[:, f] = rng.integers(0, n_cats[f], size=n_true)
                m = self._score(true_rows, torch.from_numpy(draw), tolerance, cat_mask).metrics
                for k, vals in accs.items():
                    if k in m:
                        vals.append(m[k])
            out["random_accuracy"] = float(np.mean(accs["reconstruction_accuracy"])) / 100.0
            if accs["continuous_accuracy"]:
                out["random_continuous_accuracy"] = float(np.mean(accs["continuous_accuracy"]))
            if accs["categorical_accuracy"]:
                out["random_categorical_accuracy"] = float(np.mean(accs["categorical_accuracy"]))
            gains("random")
        return out

    def _baseline_accuracy(
        self,
        true_rows: Tensor,
        tolerance: Tensor,
        cat_mask: Tensor,
    ) -> float:
        """Accuracy of a reconstruction that knows only the column marginals of ``reference_rows``.

        Each baseline row is built by drawing every column independently from that column's
        empirical distribution in ``reference_rows`` — never from the attacked batch — so it uses
        no leaked signal at all. Scored through exactly the same matching pipeline so the
        comparison is apples to apples.
        """
        rng = np.random.default_rng(self.baseline_seed)
        n_true, n_features = true_rows.shape
        observed = self._reference_scalar().numpy()

        accuracies: list[float] = []
        for _ in range(max(self.baseline_draws, 1)):
            draw = np.empty((n_true, n_features), dtype=observed.dtype)
            for f in range(n_features):
                draw[:, f] = rng.choice(observed[:, f], size=n_true, replace=True)
            scored = self._score(true_rows, torch.from_numpy(draw).float(), tolerance, cat_mask)
            accuracies.append(scored.metrics["reconstruction_accuracy"])
        return float(np.mean(accuracies))

    def _reidentification_rate(
        self,
        true_rows: Tensor,
        recon_rows: Tensor,
        tolerance: Tensor,
        cat_mask: Tensor,
    ) -> float:
        """Fraction of victim rows whose nearest reconstructed row recovers them exactly.

        Distinct from ``exact_row_recovery``: this asks whether an adversary
        holding the reconstruction could *point at* a specific victim record,
        using nearest-neighbour lookup rather than an oracle assignment.  It is
        the metric that corresponds to a re-identification attack.
        """
        if recon_rows.shape[0] == 0:
            return 0.0
        stds = self.schema.scalar_col_stds(true_rows).clamp_min(1e-12)
        t_norm = true_rows / stds
        r_norm = recon_rows / stds
        nearest = torch.cdist(t_norm, r_norm).argmin(dim=1)          # [T]
        hits = _hit_matrix(true_rows, recon_rows, tolerance, cat_mask)
        exact = hits.all(dim=-1)                                      # [T, R]
        return exact[torch.arange(true_rows.shape[0]), nearest].float().mean().item()


def _relative_gain(accuracy: float, baseline: float) -> float:
    """Headroom-normalised improvement over a prior-only reconstruction.

    Both inputs are on the 0–100 scale.  Returns 0.0 when the baseline already
    saturates, so a degenerate table cannot manufacture an impressive number.
    """
    headroom = 100.0 - baseline
    if headroom <= 1e-9:
        return 0.0
    return (accuracy - baseline) / headroom


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
