#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Validation pass over a finished CIFAR frontier campaign.

The campaign optimizes TPR @ FPR = 1%, which is a proxy. This script answers the
two questions that leaves open, using a stronger attack (more reference models)
on a larger audit set:

1. What is the risk at the FPR levels we report (default 0.1% and 1%)?
   Measured on the Pareto points only.
2. Was the proxy legitimate? Spearman correlation between ranking configs by
   TPR@1% and by TPR@0.1%, over configs spanning the whole observed range.

Usage:
    python validate_frontier.py --n-refs 8 --audit-size 10000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_campaign import N_TARGET_TRAIN, load_splits, make_recipe  # noqa: E402

from leakpro.optimization import (  # noqa: E402
    AttackScores,
    EvaluationRecord,
    PETRecipe,
    confidence_signal,
    proxy_agreement,
    train_with_dpsgd,
    validate_frontier,
)
from leakpro.optimization.validation import resolution_warning  # noqa: E402
from leakpro.utils.logger import logger  # noqa: E402


def make_revalidate_fn(splits: dict, recipe: PETRecipe, n_refs: int, device: str,
                       audit_size: int, seed: int = 99) -> callable:
    """Stronger attack: more matched references, larger audit set, fresh reference seed.

    Reference models still mimic the candidate configuration fully; only the
    attack's statistical power changes relative to the loop.
    """
    rng_master = np.random.default_rng(seed)
    members = splits["audit_members_large"][:audit_size]
    nonmembers = splits["audit_nonmembers_large"][:audit_size]
    logger.info(f"Validation audit set: {len(members)} members / {len(nonmembers)} nonmembers, {n_refs} references.")

    def revalidate_fn(config: dict) -> AttackScores:
        x, y = splits["x"], splits["y"]
        kind = recipe.output_kind
        target = train_with_dpsgd(recipe, config, splits["target_train"], device)

        ref_phi_m = np.zeros(len(members))
        ref_phi_n = np.zeros(len(nonmembers))
        rng = np.random.default_rng(rng_master.integers(1 << 30))
        for _ in range(n_refs):
            sub = rng.choice(splits["ref_pool"], size=N_TARGET_TRAIN, replace=False)
            ref = train_with_dpsgd(recipe, config, sub, device)
            ref_phi_m += confidence_signal(ref, x[members], y[members], device, kind) / n_refs
            ref_phi_n += confidence_signal(ref, x[nonmembers], y[nonmembers], device, kind) / n_refs
            del ref
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        scores = AttackScores(
            member_scores=confidence_signal(target, x[members], y[members], device, kind) - ref_phi_m,
            nonmember_scores=confidence_signal(target, x[nonmembers], y[nonmembers], device, kind) - ref_phi_n,
        )
        del target
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return scores

    return revalidate_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", default="leakpro_output/pet_optimization")
    parser.add_argument("--epochs", type=int, default=15, help="must match the campaign's epochs")
    parser.add_argument("--n-refs", type=int, default=8, help="reference models (the loop used 2)")
    parser.add_argument("--audit-size", type=int, default=10000, help="members = nonmembers = this many")
    parser.add_argument("--report-fprs", type=float, nargs="+", default=[0.001, 0.01])
    parser.add_argument("--agreement-configs", type=int, default=6, help="0 disables the proxy check")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="must match the campaign's seed")
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir)
    records = [EvaluationRecord(json.loads(line))
               for line in (campaign_dir / "evaluations.jsonl").read_text().strip().splitlines()]
    logger.info(f"Loaded {len(records)} evaluations from {campaign_dir}.")

    # Validation only means anything if it retrains with the campaign's own
    # recipe: a different seed reshuffles every split, a different epoch count
    # is a different model. Cross-check against what the campaign recorded
    # instead of trusting the flags.
    meta_path = campaign_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for field in ("epochs", "seed"):
            if field in meta and getattr(args, field) != meta[field]:
                raise SystemExit(
                    f"--{field}={getattr(args, field)} does not match the campaign's {field}={meta[field]} "
                    f"(from {meta_path}). Validating with a mismatched recipe produces confidently wrong numbers."
                )
    else:
        record_seeds = {r.get("seed") for r in records if "seed" in r}
        if record_seeds and record_seeds != {args.seed}:
            raise SystemExit(
                f"--seed={args.seed} does not match the seed(s) {sorted(record_seeds)} stored in the "
                "campaign records. Validating with a mismatched seed reshuffles every split."
            )
        logger.warning("No run_meta.json in the campaign dir; --epochs cannot be cross-checked. "
                       "Make sure it matches the campaign's epochs.")

    splits = load_splits(seed=args.seed, audit_size=args.audit_size)
    warning = resolution_warning(args.audit_size, min(args.report_fprs))
    if warning:
        logger.warning(warning)

    recipe = make_recipe(splits, args.epochs)
    revalidate_fn = make_revalidate_fn(splits, recipe, args.n_refs, args.device, args.audit_size)

    start = time.time()
    validated = validate_frontier(records, revalidate_fn, report_fprs=tuple(args.report_fprs))
    out = {"validated_frontier": validated}

    if args.agreement_configs > 0:
        out["proxy_agreement"] = proxy_agreement(
            records, revalidate_fn,
            proxy_fpr=records[0].get("proxy_fpr", 0.01),
            target_fpr=min(args.report_fprs),
            n_configs=args.agreement_configs,
        )

    path = campaign_dir / "validation.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info(f"Validation done in {time.time() - start:.0f}s. Written to {path}.")


if __name__ == "__main__":
    main()
