#!/usr/bin/env python3
import contextlib
import itertools
import subprocess
from pathlib import Path

import yaml  # pip install pyyaml

AUDIT_YAML = Path("audit.yaml")   # will be overwritten each run
LEAKPRO    = Path("run_leakpro.py")   # your runner
PYTHON_BIN = "python"             # or your venv python

def deep_copy(obj):
    return yaml.safe_load(yaml.safe_dump(obj, sort_keys=False))

def main() -> None:
    # load base once
    base_cfg = yaml.safe_load(AUDIT_YAML.read_text(encoding="utf-8"))

    # grids
    lrs     = [0.0002, 0.002]          # gen_lr = dis_lr
    n_dises = [1, 2, 4]
    bszs    = [60, 250, 500]
    # alphas  = [0.2, 0.5, 0.8, 1.0]
    alphas  = [0]

    combos = list(itertools.product(lrs, n_dises, bszs, alphas))

    for lr, n_dis, bsz, alpha in combos:
        cfg = deep_copy(base_cfg)

        # assume first attack block is your CTGAN/PLGMI
        atk = cfg["audit"]["attack_list"][0]
        atk["gen_lr"]     = float(lr)
        atk["dis_lr"]     = float(lr)
        atk["n_dis"]      = int(n_dis)
        atk["batch_size"] = int(bsz)
        atk["alpha"]      = float(alpha)

        # overwrite audit.yaml
        AUDIT_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        cmd = [PYTHON_BIN, str(LEAKPRO), "--config", str(AUDIT_YAML)]
        with contextlib.suppress(subprocess.CalledProcessError):
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
