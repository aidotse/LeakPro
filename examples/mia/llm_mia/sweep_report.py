"""Combine the results of several LeakPro runs (e.g. full vs LoRA fine-tuning) into one report.

    python sweep_report.py leakpro_output_full leakpro_output_lora --out sweep_report

Each run directory is the ``audit.output_dir`` of one ``run_audit.py`` invocation. Every saved result
(``data_objects/*.json``) is loaded with ``MIAResult.load`` and tagged with the run's label (the
directory suffix after ``leakpro_output_``, or ``--labels``). ``MIAResult.create_results`` then draws
one combined log-log ROC and a LaTeX table of AUC and TPR at fixed FPRs — no new metric code,
just the existing result API over more than one run.
"""

import argparse
from pathlib import Path

from leakpro.reporting.mia_result import MIAResult


def load_run(run_dir: Path, label: str) -> list:
    """Load every MIAResult saved by one run and stamp the run label into its id (legend/table name)."""
    results = []
    for path in sorted((run_dir / "data_objects").glob("*.json")):
        res = MIAResult.load(str(path))
        # reduce_to_unique_labels() splits the id on "-" and uses the first part as the series name.
        name = res.result_name.replace("-", "") + f"[{label}]"
        config_hash = res.id.split("-")[-1]
        res.id = f"{name}-{config_hash}"
        res.result_name = name
        results.append(res)
    if not results:
        raise FileNotFoundError(f"no results under {run_dir / 'data_objects'}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="audit.output_dir of each run")
    ap.add_argument("--labels", nargs="*", default=None, help="One label per run (default: dir suffix)")
    ap.add_argument("--out", default="sweep_report")
    args = ap.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    labels = args.labels or [d.name.replace("leakpro_output_", "") for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise ValueError("--labels must have one entry per run")

    results = []
    for run_dir, label in zip(run_dirs, labels):
        results.extend(load_run(run_dir, label))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latex = MIAResult.create_results(results=results, save_dir=str(out))
    (out / "sweep_results.tex").write_text(latex)

    print(f"{'result':<28}{'AUC':>8}   fixed-FPR table")
    for res in results:
        auc = f"{res.roc_auc:.4f}" if res.roc_auc is not None else "  n/a"
        print(f"{res.id.split('-')[0]:<28}{auc:>8}   {res.fixed_fpr_table}")
    print(f"combined ROC: {out / 'ROC.png'}   LaTeX: {out / 'sweep_results.tex'}")


if __name__ == "__main__":
    main()
