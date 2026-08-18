#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Detached entry point for a webapp PET optimization run.

Run as a subprocess, never on the API's thread pool. A campaign is
``n_configs x (1 + n_refs)`` full trainings, which is minutes for a logistic
regression and hours for anything convolutional, and the API demotes any job
still marked running to failed when it restarts. Keeping the work in its own
process means a restart costs nothing: ``Campaign`` appends one JSON line per
evaluated setting and resumes from that file, so killing the process is also
how cancellation works.

Usage:
    python -m leakpro.webapp.backend.pet_runner <job_dir> <model_name> [--verify N]
"""

import argparse
import importlib.util
import json
import math
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

# Set before torch touches CUDA. The spawner sets this too, but the runner is a
# fresh process every launch, so setting it here means the fix cannot be defeated
# by a stale (un-reloaded) backend. Lets CUDA grow its allocation instead of
# pre-reserving fixed segments, which is the usual cause of a spurious OOM when
# the campaign trains many models back to back.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from leakpro.optimization import (
    Campaign,
    KnobSpace,
    build_campaign_fns,
    default_dpsgd_space,
    pareto_front,
    resolution_warning,
    tpr_at_fpr,
)
from leakpro.optimization.adapters import carve_splits, detect_binary, recipe_from_module
from leakpro.utils.logger import logger

DEFAULT_N_CONFIGS = 24
DEFAULT_N_REFS = 2
DEFAULT_EPOCHS = 10
# Per-sample-gradient memory cap under DP-SGD. 256 fits a small model but OOMs a
# ResNet on a 24 GB card; 32 is the safe webapp default. Speed only, not accounting.
DEFAULT_MAX_PHYSICAL_BATCH = 32

# The verification pass exists because the search runs a deliberately cheap
# attack. It re-attacks one setting with more reference models so the number
# the user adopts is not the one the search guessed with.
VERIFY_N_REFS = 6

# The false-alarm rate every reported figure uses, search and verification
# alike, so the two are directly comparable.
PROXY_FPR = 0.01


def pet_dir(job_dir: Path, model_name: str) -> Path:
    """Where one model's campaign keeps its evaluations and status."""
    return job_dir / "pet" / model_name


def json_safe(value: Any) -> Any:  # noqa: ANN401
    """Make a record safe for `JSON.parse`.

    Python writes a bare ``Infinity`` for ``float("inf")``, which every browser
    rejects. The non-private anchor records exactly that as its epsilon, so
    without this the frontend cannot read the file at all.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return json_safe(value.item())
    return value


def write_status(out: Path, **fields: Any) -> None:
    """Overwrite status.json; the API reads this rather than tracking the process."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(json_safe(fields), indent=2))


def _register_dataset_handler(job_dir: Path) -> None:
    """Make a user-uploaded dataset_handler.py importable under its original name.

    Population pickles reference the class path they were written with, so
    unpickling fails unless that module name resolves. The audit worker solves
    this the same way; the campaign has to, because it loads the same files.
    """
    path = job_dir / "dataset_handler.py"
    if not path.exists():
        return
    spec = importlib.util.spec_from_file_location("dataset_handler", path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules.setdefault("dataset_handler", module)

    # CelebADataHandler -> celebA_data_handler, CifarDataHandler -> cifar_data_handler
    for attr in dir(module):
        obj = getattr(module, attr, None)
        if isinstance(obj, type) and hasattr(obj, "UserDataset") and attr != "AbstractInputHandler":
            original = attr.replace("DataHandler", "_data_handler").replace("InputHandler", "_input_handler")
            original = original[0].lower() + original[1:]
            sys.modules.setdefault(original, module)
            logger.info(f"Registered dataset_handler.py as '{original}' for unpickling.")


def _load_dataset(data_path: str, job_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    import joblib

    _register_dataset_handler(job_dir)
    # The job directory itself may hold the module a dataset was pickled from.
    if str(job_dir) not in sys.path:
        sys.path.insert(0, str(job_dir))
    dataset = joblib.load(data_path)
    data = getattr(dataset, "data", None)
    targets = getattr(dataset, "targets", None)
    if data is None or targets is None:
        raise ValueError(f"{data_path} has no .data/.targets; a campaign needs an indexable tensor dataset.")
    x = data if isinstance(data, torch.Tensor) else torch.as_tensor(np.asarray(data))
    y = targets if isinstance(targets, torch.Tensor) else torch.as_tensor(np.asarray(targets))
    return x.float(), y


def _init_params(model_dir: Path) -> dict:
    meta_path = model_dir / "model_metadata.pkl"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("rb") as f:
            meta = pickle.load(f)  # noqa: S301
    except Exception:  # noqa: BLE001
        return {}
    params = getattr(meta, "init_params", None)
    if params is None and isinstance(meta, dict):
        params = meta.get("init_params")
    return dict(params or {})


def build(job: dict, job_dir: Path, model_name: str) -> tuple:
    """Assemble (recipe, splits, knob_space, settings) from persisted job state."""
    model = next((m for m in job.get("models", []) if m["name"] == model_name), None)
    if model is None:
        raise KeyError(f"Job has no model named '{model_name}'.")

    data_path = job.get("data_path")
    if not data_path or not Path(data_path).exists():
        raise FileNotFoundError("The job's dataset is missing; a campaign retrains from it.")

    arch = job_dir / "arch.py"
    if not arch.exists():
        raise FileNotFoundError("arch.py is missing; a campaign needs the architecture to retrain.")

    x, y = _load_dataset(data_path, job_dir)
    train_params = model.get("train_params") or {}
    model_class = model.get("model_class") or job.get("model_class")
    init_params = _init_params(job_dir / "models" / model_name)

    # The head shape decides the loss, the label dtype and how the attack reads
    # confidences, so ask the model rather than guessing from the class count.
    binary = detect_binary(arch, x, model_class, init_params)
    y = y.float() if binary else y.long()

    recipe = recipe_from_module(
        module_path=arch,
        x=x,
        y=y,
        epochs=int(train_params.get("epochs") or DEFAULT_EPOCHS),
        model_class=model_class,
        init_params=init_params,
        optimizer_name=str(train_params.get("optimizer") or "adam").lower(),
        binary=binary,
    )
    splits = carve_splits(x, y, seed=int(job.get("seed", 0)))
    return recipe, splits, default_dpsgd_space(), model


def _apply_advanced(space: KnobSpace, advanced: dict) -> KnobSpace:
    """Pin any knob the user typed a value for; the rest stay searched."""
    for name, value in (advanced or {}).items():
        if value is None:
            continue
        try:
            space = space.fix(name, float(value))
        except KeyError:
            logger.warning(f"Ignoring unknown knob '{name}'.")
    return space


def _is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory error, however this torch version raises it."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def _run_with_oom_backoff(build_fns, run, max_physical_batch: int, floor: int = 4):  # noqa: ANN001
    """Run `run(train_fn, utility_fn, attack_fn)`, halving the batch cap on OOM.

    The cap is a memory measure only, so shrinking it never changes a result.
    The campaign resumes from its JSONL, so a retry re-runs only the config that
    ran out of memory, not the whole sweep.
    """
    batch = max_physical_batch
    while True:
        try:
            return run(*build_fns(batch))
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc) or batch <= floor:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch = max(floor, batch // 2)
            logger.warning(f"CUDA out of memory; retrying at max_physical_batch={batch}.")


def run_campaign(job_dir: Path, model_name: str) -> None:
    """Run the whole sweep, then record the best trade-offs."""
    out = pet_dir(job_dir, model_name)
    job = json.loads((job_dir / "state.json").read_text())
    request = json.loads((out / "request.json").read_text()) if (out / "request.json").exists() else {}
    n_configs = int(request.get("n_configs") or DEFAULT_N_CONFIGS)
    n_refs = int(request.get("n_refs") or DEFAULT_N_REFS)
    max_physical_batch = int(request.get("max_physical_batch") or DEFAULT_MAX_PHYSICAL_BATCH)

    recipe, splits, space, model = build(job, job_dir, model_name)
    space = _apply_advanced(space, request.get("advanced") or {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Say so up front when the audit set cannot resolve the reported FPR:
    # a frontier drawn from a handful of events looks identical to a real one.
    warning = resolution_warning(len(splits["audit_nonmembers"]), PROXY_FPR)
    if warning:
        logger.warning(warning)

    write_status(out, status="running", model_name=model_name, n_configs=n_configs,
                 baseline_utility=model.get("test_accuracy"), resolution_warning=warning)

    delta = float(request.get("delta") or 1e-5)
    seed = int(job.get("seed", 0))

    def build_fns(batch: int):  # noqa: ANN202
        return build_campaign_fns(
            recipe, splits, n_refs=n_refs, device=device,
            utility_metric="accuracy", delta=delta, max_physical_batch=batch,
        )

    def run_campaign_fns(train_fn, utility_fn, attack_fn):  # noqa: ANN001, ANN202
        campaign = Campaign(train_fn, utility_fn, attack_fn, knob_space=space, output_dir=out, seed=seed)
        return campaign.run(n_configs)

    records = _run_with_oom_backoff(build_fns, run_campaign_fns, max_physical_batch)

    front = pareto_front(records)
    write_status(
        out, status="done", model_name=model_name, n_configs=n_configs,
        baseline_utility=model.get("test_accuracy"),
        best_indices=[r["index"] for r in front],
        resolution_warning=warning,
    )
    logger.info(f"Campaign finished: {len(records)} settings, {len(front)} on the frontier.")


def run_verification(job_dir: Path, model_name: str, index: int) -> None:
    """Re-measure one setting with a stronger attack than the search used."""
    out = pet_dir(job_dir, model_name)
    job = json.loads((job_dir / "state.json").read_text())
    target = out / f"verification_{index}.json"

    records = [json.loads(line) for line in (out / "evaluations.jsonl").read_text().splitlines() if line.strip()]
    record = next((r for r in records if r["index"] == index), None)
    if record is None:
        raise KeyError(f"No evaluated setting with index {index}.")

    estimated = {"attack_tpr": record.get("attack_tpr"), "utility": record["utility"]}
    target.write_text(json.dumps(json_safe({"status": "running", "index": index, "estimated": estimated}), indent=2))

    request = json.loads((out / "request.json").read_text()) if (out / "request.json").exists() else {}
    max_physical_batch = int(request.get("max_physical_batch") or DEFAULT_MAX_PHYSICAL_BATCH)
    recipe, splits, _, _ = build(job, job_dir, model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = record["config"]

    def build_fns(batch: int):  # noqa: ANN202
        return build_campaign_fns(
            recipe, splits, n_refs=VERIFY_N_REFS, device=device,
            utility_metric="accuracy", max_physical_batch=batch,
        )

    def verify_once(train_fn, utility_fn, attack_fn):  # noqa: ANN001, ANN202
        model = train_fn(config)
        util = float(utility_fn(model))
        tpr_val, _, _ = tpr_at_fpr(attack_fn(model, config), record.get("proxy_fpr", 0.01))
        return util, tpr_val, getattr(model, "campaign_extras", {}).get("epsilon")

    utility, tpr, epsilon = _run_with_oom_backoff(build_fns, verify_once, max_physical_batch)

    target.write_text(json.dumps(json_safe({
        "status": "done",
        "index": index,
        "estimated": estimated,
        "verified": {
            "attack_tpr": tpr,
            "utility": utility,
            "epsilon": epsilon,
        },
    }), indent=2))
    logger.info(f"Verified setting {index}: TPR {tpr:.4f} (search estimated {estimated['attack_tpr']}).")


def main() -> None:
    """Dispatch to a campaign or a single verification, recording failures either way."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir")
    parser.add_argument("model_name")
    parser.add_argument("--verify", type=int, default=None)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    out = pet_dir(job_dir, args.model_name)
    try:
        if args.verify is None:
            run_campaign(job_dir, args.model_name)
        else:
            run_verification(job_dir, args.model_name, args.verify)
    except Exception as exc:  # noqa: BLE001
        detail = f"{exc}\n{traceback.format_exc()}"
        logger.error(f"PET run failed: {detail}")
        if args.verify is None:
            write_status(out, status="failed", model_name=args.model_name, error=str(exc))
        else:
            (out / f"verification_{args.verify}.json").write_text(
                json.dumps({"status": "failed", "index": args.verify,
                            "estimated": {"attack_tpr": None, "utility": None}, "error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
