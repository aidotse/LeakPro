"""Audit a fine-tuned causal LM with the LLM attacks in audit.yaml.

    python run_audit.py                          # audits target.target_folder from audit.yaml
    python run_audit.py --target ./target_lora   # audit a different prepared target (sweep)

Same shape as the other examples: LeakPro(data handler, audit.yaml, model_handler=...).run_audit().
"""

import argparse
import tempfile
from pathlib import Path

import yaml

from llm_data_handler import LLMDataHandler
from llm_model_handler import LLMModelHandler

from leakpro import LeakPro


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="audit.yaml")
    ap.add_argument("--target", default=None, help="Override target.target_folder; output_dir becomes leakpro_output_<name>")
    ap.add_argument("--pdf", action="store_true", help="Also compile the LaTeX report")
    args = ap.parse_args()

    audit_path = args.audit
    if args.target:
        with open(args.audit) as f:
            cfg = yaml.safe_load(f)
        target = Path(args.target)
        cfg["target"]["target_folder"] = str(target)
        cfg["audit"]["output_dir"] = f"./leakpro_output_{target.name.replace('target_', '')}"
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp.close()
        audit_path = tmp.name

    leakpro = LeakPro(LLMDataHandler, audit_path, model_handler=LLMModelHandler)
    results = leakpro.run_audit(create_pdf=args.pdf)
    for res in results:
        print(f"{res.result_name}: AUC {res.roc_auc:.4f}  " + "  ".join(f"{k} {v}" for k, v in res.fixed_fpr_table.items()))


if __name__ == "__main__":
    main()
