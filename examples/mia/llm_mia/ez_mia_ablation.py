#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""One-toggle-at-a-time ablation of EZ-MIA's edge-case scoring on one fixed target.

    cd ez-mia
    python ../ez_mia_ablation.py                           # WikiText: audit.yaml -> target_full
    python ../ez_mia_ablation.py --audit audit_xsum.yaml   # XSum

The target and reference forward passes run ONCE; every arm below re-scores that same evidence, so
the arms differ only in the scoring rule -- the trained target, the audited rows and the reference are
identical, and scoring is deterministic. Two toggles, 2x2:

  policy           "shipped"   -- ez_scores as committed: zero error positions (paper E.5) and N == 0
                                  (E.6, ratio only) rank as members; 1 <= n_err < min_error_positions
                                  scores 0.0.
                   "reference" -- the paper's released code: every row with N == 0 or
                                  n_err < min_error_positions scores 0.0; nothing is forced.
  ignore_first_position  True (shipped default) / False

Arm A is the shipped default. B toggles only the policy, C only position 0, D both. Per arm it reports
AUC and TPR at fixed FPRs (via MIAResult, i.e. the same metric code as an audit), the forced/floored row
counts, and how many rows' scores differ from arm A. Only `aggregation: ratio` is supported -- it is the
paper's score and the only one the reference policy is defined for.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from llm_data_handler import LLMDataHandler
from llm_model_handler import LLMModelHandler

from leakpro import LeakPro
from leakpro.attacks.mia_attacks.llm.ez_mia import AttackEZMIA, ez_scores
from leakpro.reporting.mia_result import MIAResult

ARMS = (  # (name, policy, ignore_first_position)
    ("A_shipped", "shipped", True),
    ("B_policy_only", "reference", True),
    ("C_position0_only", "shipped", False),
    ("D_both", "reference", False),
)
FPRS = ("TPR@0.1%FPR", "TPR@1%FPR", "TPR@10%FPR")


def reference_policy_scores(delta: np.ndarray, error: np.ndarray, min_error_positions: int,
                            ignore_first_position: bool) -> tuple:
    """P/N with every N == 0 or n_err < min_error_positions row floored to 0.0, nothing forced."""
    error = error.copy()
    if ignore_first_position and error.shape[1] > 0:
        error[:, 0] = False
    sel = np.where(error, delta, 0.0)
    P = np.clip(sel, 0.0, None).sum(axis=1)  # noqa: N806
    N = np.abs(np.clip(sel, None, 0.0)).sum(axis=1)  # noqa: N806
    n_err = error.sum(axis=1)
    floored = (N == 0) | (n_err < min_error_positions)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(floored, 0.0, P / np.where(N == 0, 1.0, N))
    return scores, floored, n_err


def main() -> None:
    """Run the forward passes once, score every arm, print and save the table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="audit.yaml")
    ap.add_argument("--out", default=None, help="CSV path (default: <audit.output_dir>/ez_mia_ablation.csv)")
    args = ap.parse_args()

    leakpro = LeakPro(LLMDataHandler, args.audit, model_handler=LLMModelHandler)
    handler = leakpro.handler
    entries = [e for e in handler.configs.audit.attack_list if e["attack"] == "ez_mia"]
    if not entries:
        raise SystemExit(f"{args.audit} has no `attack: ez_mia` entry")
    attack = AttackEZMIA(handler, {k: v for k, v in entries[0].items() if k != "attack"})
    if attack.configs.aggregation != "ratio":
        raise SystemExit(f"only aggregation: ratio is supported, got {attack.configs.aggregation}")

    attack.prepare_attack()  # the only expensive step: target + reference forward passes
    labels = attack._audit_labels
    target, reference = attack.evidence_set.target, attack.evidence_set.ref(0)
    delta = target.logprob - reference.logprob
    error = (target.argmax != target.token_ids) & target.mask
    min_err = attack.configs.min_error_positions

    rows, base_scores = [], None
    for name, policy, ignore_first in ARMS:
        if policy == "shipped":
            res = ez_scores(delta, error, "ratio", min_error_positions=min_err, ignore_first_position=ignore_first)
            scores = res.scores
            err = error.copy()
            if ignore_first:
                err[:, 0] = False
            n_err = err.sum(axis=1)
            counts = {"forced_zero_errors_E5": int((res.forced & (n_err == 0)).sum()),
                      "forced_N0_E6": int((res.forced & (n_err > 0)).sum()),
                      "scored_0_insufficient": int(res.insufficient.sum()),
                      "scored_0_reference_floor": 0}
        else:
            scores, floored, n_err = reference_policy_scores(delta, error, min_err, ignore_first)
            counts = {"forced_zero_errors_E5": 0, "forced_N0_E6": 0, "scored_0_insufficient": 0,
                      "scored_0_reference_floor": int(floored.sum())}
        if base_scores is None:
            base_scores = scores
        result = MIAResult.from_full_scores(true_membership=labels, signal_values=scores, result_name=name)
        rows.append({
            "arm": name, "policy": policy, "ignore_first_position": ignore_first,
            "auc": round(float(result.roc_auc), 4),
            **{k: round(float(result.fixed_fpr_table[k]), 4) for k in FPRS},
            **counts,
            "zero_error_rows": int((n_err == 0).sum()),
            "rows_scored_differently_from_A": int((~np.isclose(scores, base_scores)).sum()),
            "n_rows": len(scores),
        })

    out = Path(args.out) if args.out else Path(handler.configs.audit.output_dir) / "ez_mia_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cols = ["arm", "auc", *FPRS, "forced_zero_errors_E5", "forced_N0_E6", "scored_0_insufficient",
            "scored_0_reference_floor", "rows_scored_differently_from_A"]
    print("  ".join(cols))
    for r in rows:
        print("  ".join(str(r[c]) for c in cols))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
