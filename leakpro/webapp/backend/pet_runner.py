"""Detached entry point for a webapp PET optimization run.

Run as a subprocess, never on the API's thread pool. A search is ``n_trials``
full trainings, each followed by an RMIA audit that trains ``n_shadow`` more
models, which is minutes for a logistic regression and hours for anything
convolutional, and the API demotes any job still marked running to failed when
it restarts. Keeping the work in its own process means a restart costs nothing:
the Optuna study (``study.db``) is the resume state, so killing the process is
also how cancellation works, and starting again continues from the last finished
trial.

Everything that scores a configuration lives in the library:
:func:`leakpro.optimization.objective.make_dpsgd_objective` trains and audits,
:func:`leakpro.optimization.optimize` searches. This module only assembles those
from a job's persisted state and writes what the API reads back off disk.

Usage:
    python -m leakpro.webapp.backend.pet_runner <job_dir> <model_name> [--verify N]
"""

from __future__ import annotations

import argparse
import inspect
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
# pre-reserving fixed segments, the usual cause of a spurious OOM when many
# models train back to back.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402

from leakpro.input_handler.user_imports import import_module_from_file  # noqa: E402
from leakpro.optimization import KnobSpace, default_dpsgd_space, optimize, pareto_trials  # noqa: E402
from leakpro.optimization.adapters import detect_binary, find_model_class, recipe_from_module  # noqa: E402
from leakpro.optimization.audit import resolved_proxy_tpr, tpr_at_fixed_fpr  # noqa: E402
from leakpro.optimization.objective import (  # noqa: E402
    AuditTarget,
    Population,
    make_dpsgd_objective,
    materialize_population,
    reaudit_trial,
    resolution_warning,
)
from leakpro.optimization.search import DEFAULT_STUDY_NAME, ObjectiveResult  # noqa: E402
from leakpro.utils.logger import logger  # noqa: E402
from leakpro.webapp.backend.dataio import load_dataset  # noqa: E402
from leakpro.webapp.backend.dataset_modules import register_job_dataset_modules  # noqa: E402
from leakpro.webapp.backend.security import load_metadata_fields  # noqa: E402

DEFAULT_N_TRIALS = 24
DEFAULT_N_SHADOW = 2
DEFAULT_EPOCHS = 10
DEFAULT_DELTA = 1e-5
# Per-sample-gradient memory cap under DP-SGD. 256 fits a small model but OOMs a
# ResNet on a 24 GB card; 32 is the safe webapp default. Speed only, not accounting.
DEFAULT_MAX_PHYSICAL_BATCH = 32
MIN_PHYSICAL_BATCH = 4
# The verification pass exists because the search runs a deliberately cheap
# audit. It re-attacks one setting's saved model with more shadow models, so the
# number the user adopts is not the one the search guessed with.
VERIFY_N_SHADOW = 6
# Population caps: members (= target training set) and held-out non-members.
# The population is 2 x MAX_TARGET at most (RMIA mimicry rule, see
# leakpro.optimization.audit); the CIFAR example uses the same sizes.
MAX_TARGET = 5000
MAX_TEST = 2000
# The false-alarm rate every reported figure uses, search and verification
# alike, so the two are directly comparable. Falls back to the coarser level
# when the held-out split cannot resolve 1% (see resolution_warning).
PROXY_FPR = 0.01
FALLBACK_FPR = 0.1

STUDY_FILE = "study.db"
STATUS_FILE = "status.json"
REQUEST_FILE = "request.json"
POPULATION_META = "population.json"


# --------------------------------------------------------------------------- #
# On-disk layout shared with the API
# --------------------------------------------------------------------------- #

def pet_dir(job_dir: Path, model_name: str) -> Path:
    """Where one model's search keeps its study, population and status."""
    return job_dir / "pet" / model_name


def json_safe(value: Any) -> Any:  # noqa: ANN401
    """Make a record safe for ``JSON.parse``.

    Python writes a bare ``Infinity`` for ``float("inf")``, which every browser
    rejects. A non-private configuration records exactly that as its epsilon, so
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


def write_status(out: Path, **fields: Any) -> None:  # noqa: ANN401
    """Overwrite status.json; the API reads this rather than tracking the process."""
    out.mkdir(parents=True, exist_ok=True)
    (out / STATUS_FILE).write_text(json.dumps(json_safe({"pid": os.getpid(), **fields}), indent=2))


def read_json(path: Path) -> dict | None:
    """Read a JSON file, or None when it is missing or caught mid-write."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def load_study(out: Path) -> optuna.study.Study | None:
    """Open a run's study read-only, or None when the run has not started."""
    db = out / STUDY_FILE
    if not db.exists():
        return None
    return optuna.load_study(study_name=DEFAULT_STUDY_NAME, storage=f"sqlite:///{db}")


def trial_records(study: optuna.study.Study | None) -> list[dict]:
    """One JSON-safe record per finished trial, in the shape the optimization view renders.

    A pruned trial (utility gate failed, or the audit could not resolve the
    operating point) keeps its utility and carries ``attack_tpr: null`` with the
    reason, so the view can show it as tested-but-unmeasurable rather than drop it.
    """
    if study is None:
        return []
    proxy_fpr = study.user_attrs.get("proxy_fpr")
    records = []
    for trial in study.trials:
        if not trial.state.is_finished() or trial.state == optuna.trial.TrialState.FAIL:
            continue
        attrs = trial.user_attrs
        if "utility" not in attrs:
            continue  # pruned before the objective ran (e.g. a crash mid-trial)
        reason = None
        if attrs.get("gated"):
            reason = "the model did not learn (utility at chance level), so it was not attacked"
        elif attrs.get("degenerate_audit"):
            reason = "every attack score tied, so no operating point could be resolved"
        records.append(json_safe({
            "index": trial.number,
            "config": attrs.get("config", trial.params),
            "utility": attrs.get("utility"),
            "attack_tpr": attrs.get("tpr"),
            "proxy_fpr": proxy_fpr,
            "attack_realized_fpr": attrs.get("realized_fpr"),
            "attack_resolution_warning": reason,
            "epsilon": attrs.get("epsilon"),
            "delta": attrs.get("delta"),
            "accountant": attrs.get("accountant"),
            "roc_auc": attrs.get("roc_auc"),
            "tuning_accounted": attrs.get("tuning_accounted", False),
        }))
    return records


# --------------------------------------------------------------------------- #
# Assembling the objective from the job's persisted state
# --------------------------------------------------------------------------- #

def model_kwargs(cls: type, num_classes: int, num_features: int | None, dpsgd: bool) -> dict:
    """Constructor kwargs for an uploaded architecture: the webapp's training rule.

    ``num_classes`` / ``num_features`` / ``dpsgd`` reach only constructors that
    accept them. A two-class problem gets a single-logit head (``num_classes=1``),
    trained with ``BCEWithLogitsLoss``, exactly as the training step does.
    """
    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}
    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = 1 if num_classes == 2 else num_classes
    if num_features is not None and "num_features" in sig.parameters:
        kwargs["num_features"] = num_features
    if dpsgd and "dpsgd" in sig.parameters:
        kwargs["dpsgd"] = True
    return kwargs


def _metadata_init_params(model_dir: Path) -> dict:
    """Constructor kwargs recorded in the model's metadata, when there is one."""
    path = model_dir / "model_metadata.pkl"
    if not path.exists():
        return {}
    try:
        meta = load_metadata_fields(path)
    except Exception:  # noqa: BLE001 - a malformed file means "no recorded params", not a crash
        return {}
    params = getattr(meta, "init_params", None)
    if params is None and isinstance(meta, dict):
        params = meta.get("init_params")
    return dict(params or {})


def _population(out: Path, dataset: Any, seed: int) -> Population:  # noqa: ANN401
    """The run's population: reused on resume, carved and written on a fresh run."""
    meta = read_json(out / POPULATION_META)
    if meta and Path(meta.get("path", "")).exists():
        return Population(**meta)
    n_target = min(len(dataset) // 2, MAX_TARGET)
    if n_target == 0:
        raise ValueError("The dataset is too small to carve a population from.")
    population = materialize_population(dataset, out, n_target=n_target, n_test=min(n_target, MAX_TEST), seed=seed)
    (out / POPULATION_META).write_text(json.dumps(population.__dict__))
    return population


def _data_modality(job: dict) -> str:
    """Audit-config modality from the job's data metadata."""
    data_type = (job.get("handler_config") or job.get("data_meta") or {}).get("data_type")
    return {"tabular": "tabular", "time_series": "timeseries"}.get(data_type, "image")


def build(job: dict, job_dir: Path, model_name: str, request: dict) -> dict:
    """Assemble everything a run needs from persisted job state.

    Returns a dict with ``recipe``, ``population``, ``target``, ``space``,
    ``proxy_fpr``, ``resolution_warning`` and the model's job entry.
    """
    model = next((m for m in job.get("models", []) if m["name"] == model_name), None)
    if model is None:
        raise KeyError(f"Job has no model named '{model_name}'.")

    data_path = job.get("data_path")
    if not data_path or not Path(data_path).exists():
        raise FileNotFoundError("The job's dataset is missing; a search retrains from it.")
    arch = job_dir / "arch.py"
    if not arch.exists():
        raise FileNotFoundError("arch.py is missing; a search needs the architecture to retrain.")

    out = pet_dir(job_dir, model_name)
    seed = int(request.get("seed") or 0)
    # The job's dataset may be pickled with the user's own class; resolve it the
    # way the audit worker does, so normalisation in its __getitem__ is kept.
    register_job_dataset_modules(job_dir)
    dataset = load_dataset(Path(data_path))
    population = _population(out, dataset, seed)
    with open(population.path, "rb") as f:
        pop = pickle.load(f)  # noqa: S301 - written by materialize_population above

    # The class the webapp trained: the one rule the training step and the
    # audit worker apply, so the search retrains what the user's model is.
    module = import_module_from_file(str(arch))
    recorded = _metadata_init_params(job_dir / "models" / model_name)
    cls = find_model_class(module, (job.get("arch_config") or {}).get("model_class"), recorded)
    sample = pop.data[:2]
    num_features = int(sample.shape[1]) if sample.ndim == 2 else None
    kwargs = {**recorded, **model_kwargs(cls, population.num_classes, num_features, dpsgd=True)}
    binary = detect_binary(arch, sample, cls.__name__, kwargs)

    train_params = model.get("train_params") or {}
    recipe = recipe_from_module(
        module_path=arch, dataset=pop,
        epochs=int(train_params.get("epochs") or DEFAULT_EPOCHS),
        model_class=cls.__name__, init_params=kwargs,
        optimizer_name=str(train_params.get("optimizer") or "adam").lower(),
        binary=binary,
    )
    target = AuditTarget(module_path=str(arch), model_class=cls.__name__,
                         dataset_name=Path(data_path).stem, data_modality=_data_modality(job),
                         init_params=kwargs or None)

    space = _apply_advanced(default_dpsgd_space(), request.get("advanced") or {})

    n_test = len(population.test_indices)
    proxy_fpr = PROXY_FPR
    warning = resolution_warning(n_test, proxy_fpr)
    if warning:
        proxy_fpr = FALLBACK_FPR
        warning = (f"The held-out split ({n_test} non-members) cannot resolve a 1% false-alarm rate, so every "
                   f"figure here is measured at {FALLBACK_FPR:.0%} instead.")
        if resolution_warning(n_test, proxy_fpr):
            raise ValueError(f"Dataset too small: {n_test} non-members cannot resolve even {FALLBACK_FPR:.0%} FPR.")

    return {"recipe": recipe, "population": population, "target": target, "space": space,
            "proxy_fpr": proxy_fpr, "resolution_warning": warning, "model": model, "seed": seed}


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


def _with_oom_backoff(make_objective, max_physical_batch: int):  # noqa: ANN001, ANN202
    """Wrap ``make_objective(cap) -> objective_fn`` so a CUDA OOM halves the cap and retries.

    The cap is a memory measure only, so shrinking it never changes a result.
    Only the configuration that ran out of memory is retried; the study keeps
    every finished trial.
    """
    state = {"cap": max_physical_batch, "objective": make_objective(max_physical_batch)}

    def objective_fn(config: dict) -> ObjectiveResult:
        while True:
            try:
                return state["objective"](config)
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc) or state["cap"] <= MIN_PHYSICAL_BATCH:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                state["cap"] = max(MIN_PHYSICAL_BATCH, state["cap"] // 2)
                state["objective"] = make_objective(state["cap"])
                logger.warning(f"CUDA out of memory; retrying at max_physical_batch={state['cap']}.")

    return objective_fn


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def run_search(job_dir: Path, model_name: str) -> None:
    """Run (or resume) the whole search, then record the best trade-offs."""
    out = pet_dir(job_dir, model_name)
    job = json.loads((job_dir / "state.json").read_text())
    request = read_json(out / REQUEST_FILE) or {}
    n_trials = int(request.get("n_trials") or DEFAULT_N_TRIALS)
    n_shadow = int(request.get("n_shadow") or DEFAULT_N_SHADOW)
    delta = float(request.get("delta") or DEFAULT_DELTA)
    max_physical_batch = int(request.get("max_physical_batch") or DEFAULT_MAX_PHYSICAL_BATCH)

    parts = build(job, job_dir, model_name, request)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if parts["resolution_warning"]:
        logger.warning(parts["resolution_warning"])

    write_status(out, status="running", model_name=model_name, n_trials=n_trials,
                 proxy_fpr=parts["proxy_fpr"], baseline_utility=parts["model"].get("test_accuracy"),
                 resolution_warning=parts["resolution_warning"])

    def make_objective(cap: int):  # noqa: ANN202
        return make_dpsgd_objective(
            parts["recipe"], parts["population"], parts["target"], out,
            proxy_fpr=parts["proxy_fpr"], delta=delta, max_physical_batch=cap,
            rmia_config={"num_shadow_models": n_shadow}, seed=parts["seed"], device=device,
        )

    study = optimize(_with_oom_backoff(make_objective, max_physical_batch), parts["space"], out,
                     n_trials=n_trials, proxy_fpr=parts["proxy_fpr"], seed=parts["seed"])

    front = pareto_trials(study)
    write_status(out, status="done", model_name=model_name, n_trials=n_trials,
                 proxy_fpr=parts["proxy_fpr"], baseline_utility=parts["model"].get("test_accuracy"),
                 best_indices=[t.number for t in front], resolution_warning=parts["resolution_warning"])
    logger.info(f"Search finished: {len(study.trials)} trials, {len(front)} on the frontier.")


def verification_path(out: Path, index: int) -> Path:
    """Where the verification of trial ``index`` is recorded."""
    return out / f"verification_{index}.json"


def run_verification(job_dir: Path, model_name: str, index: int) -> None:
    """Re-audit one trial's saved model with more shadow models than the search used."""
    out = pet_dir(job_dir, model_name)
    job = json.loads((job_dir / "state.json").read_text())
    request = read_json(out / REQUEST_FILE) or {}
    target_path = verification_path(out, index)

    study = load_study(out)
    trial = next((t for t in (study.trials if study else []) if t.number == index), None)
    if trial is None or "trial_dir" not in trial.user_attrs:
        raise KeyError(f"No evaluated setting with index {index}.")
    attrs = trial.user_attrs
    estimated = {"attack_tpr": attrs.get("tpr"), "utility": attrs.get("utility")}
    target_path.write_text(json.dumps(json_safe({"status": "running", "index": index, "estimated": estimated}),
                                      indent=2))

    parts = build(job, job_dir, model_name, request)
    proxy_fpr = float(study.user_attrs.get("proxy_fpr", parts["proxy_fpr"]))
    result = reaudit_trial(attrs["trial_dir"], population_path=parts["population"].path, target=parts["target"],
                           rmia_config={"num_shadow_models": VERIFY_N_SHADOW}, seed=parts["seed"])
    tpr, realized_fpr, degenerate = resolved_proxy_tpr(result, proxy_fpr)

    target_path.write_text(json.dumps(json_safe({
        "status": "done",
        "index": index,
        "estimated": estimated,
        "verified": {
            "attack_tpr": tpr,
            "attack_tpr_table": None if degenerate else tpr_at_fixed_fpr(result, proxy_fpr),
            # The model is not retrained, so its utility and epsilon are the trial's.
            "utility": attrs.get("utility"),
            "epsilon": attrs.get("epsilon"),
            "n_shadow": VERIFY_N_SHADOW,
            # The operating point behind the verified number. Without it the UI
            # cannot tell a resolved measurement from one the audit set was too
            # coarse to resolve; both arrive as a bare TPR.
            "realized_fpr": realized_fpr,
            "resolution_warning": ("every attack score tied, so no operating point could be resolved"
                                   if degenerate else None),
        },
    }), indent=2))
    logger.info(f"Verified setting {index}: TPR {tpr} at realized FPR {realized_fpr} "
                f"(search estimated {estimated['attack_tpr']}).")


def main() -> None:
    """Dispatch to a search or a single verification, recording failures either way."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir")
    parser.add_argument("model_name")
    parser.add_argument("--verify", type=int, default=None)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    out = pet_dir(job_dir, args.model_name)
    try:
        if args.verify is None:
            run_search(job_dir, args.model_name)
        else:
            run_verification(job_dir, args.model_name, args.verify)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"PET run failed: {exc}\n{traceback.format_exc()}")
        if args.verify is None:
            write_status(out, status="failed", model_name=args.model_name, error=str(exc))
        else:
            verification_path(out, args.verify).write_text(json.dumps({
                "status": "failed", "index": args.verify,
                "estimated": {"attack_tpr": None, "utility": None}, "error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
