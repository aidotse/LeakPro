#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Validation pass over a finished CIFAR DP-SGD optimization run.

The loop optimizes TPR @ FPR = 1% with a modest number of RMIA reference models.
This script re-audits each Pareto configuration with *more* reference models — a
stronger attack — reusing the target already trained during the optimization, and
reports TPR at the FPR levels you actually care about (default 0.1% and 1%)
straight from RMIA's ``fixed_fpr_table``. It also checks that ranking configs by
TPR@1% agrees with ranking them by TPR@0.1% (``proxy_agreement``).

Usage:
    python validate_frontier.py --run-dir leakpro_output/dpsgd_optimization --n-shadow 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_optimization import _config_dir, _write_audit_yaml  # noqa: E402

from leakpro.optimization.search import DEFAULT_STUDY_NAME  # noqa: E402
from leakpro.optimization import (  # noqa: E402
    proxy_agreement,
    run_rmia_audit,
    validate_frontier,
)
from dp_handler import CifarDPHandler  # noqa: E402, I001
from leakpro.utils.logger import logger  # noqa: E402


def make_revalidate_fn(run_dir: Path, pop_path: str, rmia_config: dict, seed: int):  # noqa: ANN201
    """Re-audit a frontier config's *already-trained* target with more RMIA references."""

    def revalidate_fn(params: dict):  # noqa: ANN202
        trial_dir = _config_dir(run_dir, params)
        target_folder = trial_dir / "target"
        if not (target_folder / "target_model.pkl").exists():
            raise FileNotFoundError(
                f"No trained target for config {params} at {target_folder}. "
                "Run the optimization with the same settings yaml first."
            )
        dpsgd_path = target_folder / "dpsgd_dic.pkl"
        val_dir = trial_dir / "validation_audit"
        config_path = trial_dir / "validation_audit.yaml"
        _write_audit_yaml(config_path, pop_path, target_folder, dpsgd_path, val_dir, rmia_config, seed)
        return run_rmia_audit(CifarDPHandler, str(config_path))

    return revalidate_fn


def main() -> None:
    """Re-audit the frontier and write validation.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="leakpro_output/dpsgd_optimization")
    parser.add_argument("--n-shadow", type=int, default=16, help="RMIA references (the loop used fewer)")
    parser.add_argument("--report-fprs", type=float, nargs="+", default=[0.001, 0.01])
    parser.add_argument("--agreement-configs", type=int, default=6, help="0 disables the proxy check")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "resolved_config.json").read_text())
    logger.info(f"Validating optimization run in {run_dir} (settings: {cfg}).")

    storage = f"sqlite:///{run_dir / 'study.db'}"
    study = optuna.load_study(study_name=DEFAULT_STUDY_NAME, storage=storage)
    pop_path = str(run_dir / "population.pkl")

    # Same RMIA settings as the loop, but with the stronger reference count.
    rmia_config = {**cfg["rmia"], "num_shadow_models": args.n_shadow}
    revalidate_fn = make_revalidate_fn(run_dir, pop_path, rmia_config, cfg["seed"])

    start = time.time()
    out = {"validated_frontier": validate_frontier(study, revalidate_fn, report_fprs=tuple(args.report_fprs))}
    if args.agreement_configs > 0:
        out["proxy_agreement"] = proxy_agreement(
            study, revalidate_fn,
            proxy_fpr=study.user_attrs.get("proxy_fpr", 0.01),
            target_fpr=min(args.report_fprs),
            n_configs=args.agreement_configs,
        )

    path = run_dir / "validation.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info(f"Validation done in {time.time() - start:.0f}s. Written to {path}.")


if __name__ == "__main__":
    main()
