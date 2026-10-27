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
    materialize_split,
    reaudit_trial,
    resolution_warning,
    tpr_uncertainty,
    train_and_audit,
)
from leakpro.optimization.search import DEFAULT_STUDY_NAME, ObjectiveResult  # noqa: E402
from leakpro.utils.logger import logger  # noqa: E402
from leakpro.webapp.backend.dataio import load_dataset  # noqa: E402
from leakpro.webapp.backend.dataset_modules import register_job_dataset_modules  # noqa: E402
from leakpro.webapp.backend.security import load_metadata_fields  # noqa: E402
from leakpro.webapp.backend.worker import _serialise_result  # noqa: E402

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
# Search-subset caps: members (= each trial's training set) and held-out
# non-members. The search ranks settings; it does not measure the user's model.
# A trial is one DP-SGD training plus n_shadow more for its audit, and at 24
# trials that is 72 trainings, so trials run on at most MAX_TARGET members:
# about five times cheaper than CIFAR-10's default 25,000-member split. The
# population is 2 x MAX_TARGET (RMIA mimicry rule, leakpro.optimization.audit);
# MAX_TEST non-members resolve a 1% FPR with about 20 false positives. The
# chosen setting is then measured at the user's own size by run_verification,
# which is where the adopted numbers come from.
MAX_TARGET = 5000
MAX_TEST = 2000
# Training settings used when neither the model's metadata nor the job's
# training step recorded one (an uploaded model with a sparse metadata file).
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 64
# The false-alarm rate every reported figure uses, search and verification
# alike, so the two are directly comparable. Falls back to the coarser level
# when the held-out split cannot resolve 1% (see resolution_warning).
PROXY_FPR = 0.01
FALLBACK_FPR = 0.1

STUDY_FILE = "study.db"
STATUS_FILE = "status.json"
REQUEST_FILE = "request.json"
POPULATION_META = "population.json"
BASELINE_FILE = "baseline.json"
# The user's-size population a verification audits, cached per model.
FULL_POPULATION_META = "population_full.json"
FULL_POPULATION_NAME = "population_full.pkl"


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
            "train_accuracy": attrs.get("train_accuracy"),
            "attack_tpr": attrs.get("tpr"),
            "attack_tpr_ci": attrs.get("attack_tpr_ci"),
            "n_members": attrs.get("n_members"),
            "n_nonmember_false_positives": attrs.get("n_nonmember_false_positives"),
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


def _metadata(model_dir: Path) -> Any:  # noqa: ANN401
    """The model's ``model_metadata.pkl`` for field inspection, or None when absent or unreadable."""
    path = model_dir / "model_metadata.pkl"
    if not path.exists():
        return None
    try:
        return load_metadata_fields(path)
    except Exception:  # noqa: BLE001 - a malformed file means "no metadata", not a crash
        return None


def _field(meta: Any, name: str, default: Any = None) -> Any:  # noqa: ANN401
    """Read a metadata field from the real schema object or from a stubbed one.

    The restricted unpickler returns a stub whose ``__dict__`` is pydantic's
    pickled state (``{"__dict__": fields, "__pydantic_fields_set__": ...}``),
    so the fields sit one level down; nested configs are stubs of the same shape.
    """
    if meta is None:
        return default
    if isinstance(meta, dict):
        return meta.get(name, default)
    value = getattr(meta, name, None)
    if value is None:
        state = getattr(meta, "__dict__", None)
        if isinstance(state, dict) and isinstance(state.get("__dict__"), dict):
            state = state["__dict__"]
        if isinstance(state, dict):
            value = state.get(name)
    return default if value is None else value


def _metadata_init_params(model_dir: Path) -> dict:
    """Constructor kwargs recorded in the model's metadata, when there is one."""
    return dict(_field(_metadata(model_dir), "init_params", {}) or {})


def _training_settings(model: dict, meta: Any) -> dict:  # noqa: ANN401
    """How the user's model was trained: epochs, optimizer, learning rate, batch size.

    The metadata is the record of what actually ran (the training step writes
    it, and an upload must carry it); the job's training request is the
    fallback for a model whose metadata could not be saved; the defaults cover
    an upload with a sparse file.
    """
    train_params = model.get("train_params") or {}
    optimizer = _field(meta, "optimizer")
    opt_params = dict(_field(optimizer, "params", {}) or {})
    loader_params = dict(_field(_field(meta, "data_loader"), "params", {}) or {})
    return {
        "epochs": int(_field(meta, "epochs") or train_params.get("epochs") or DEFAULT_EPOCHS),
        "optimizer": str(_field(optimizer, "name") or train_params.get("optimizer") or "adam").lower(),
        "learning_rate": float(opt_params.get("lr") or train_params.get("learning_rate") or DEFAULT_LEARNING_RATE),
        "batch_size": int(loader_params.get("batch_size") or train_params.get("batch_size") or DEFAULT_BATCH_SIZE),
    }


def _assert_classification(dataset: Any) -> None:  # noqa: ANN401
    """Refuse a dataset whose targets are not class labels.

    The population writer casts targets with ``.long()`` and the objective
    scores accuracy, so a regression or forecasting target would be truncated
    to integers and the search would run on garbage without a single error.
    """
    targets = getattr(dataset, "targets", None)
    if targets is None:
        # No label array to read: one pass over the labels, never a prefix (a
        # label-sorted dataset would read as a single class).
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
        targets = torch.cat([yb.reshape(-1) for _, yb in loader])
    values = np.asarray(targets.cpu().numpy() if torch.is_tensor(targets) else targets).reshape(-1)
    if values.dtype.kind == "f" and not np.all(np.isclose(values, np.round(values))):
        raise ValueError("This job's targets are continuous values, not class labels. Protection settings can only "
                         "be searched for a classification model; regression and forecasting are not supported yet.")
    if values.dtype.kind not in "iufb":
        raise ValueError(f"Targets of dtype {values.dtype} are not class labels.")
    if len(np.unique(values)) < 2:
        raise ValueError("The dataset has a single class; there is nothing to classify or to protect.")


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
    _assert_classification(dataset)
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
    # The class count is the dataset's, not the subset's: a rare class missing
    # from the subset would otherwise shrink the head below the user's model.
    # The recorded constructor kwargs win over the derived ones; only the
    # dpsgd flag is forced, since the retrain is private.
    job_classes = (job.get("handler_config") or job.get("data_meta") or {}).get("n_classes")
    derived = model_kwargs(cls, int(job_classes or population.num_classes), num_features, dpsgd=True)
    kwargs = {**derived, **recorded, **({"dpsgd": True} if "dpsgd" in derived else {})}
    binary = detect_binary(arch, sample, cls.__name__, kwargs)

    training = _training_settings(model, _metadata(job_dir / "models" / model_name))
    recipe = recipe_from_module(
        module_path=arch, dataset=pop,
        epochs=training["epochs"],
        model_class=cls.__name__, init_params=kwargs,
        optimizer_name=training["optimizer"],
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
            "proxy_fpr": proxy_fpr, "resolution_warning": warning, "model": model, "seed": seed,
            "dataset": dataset, "training": training}


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

    # Keeps the trial_number keyword, so optimize() still keys each trial's
    # directory on its number through the wrapper.
    def objective_fn(config: dict, trial_number: int | None = None) -> ObjectiveResult:
        while True:
            try:
                return state["objective"](config, trial_number=trial_number)
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

def _baseline_config(training: dict) -> dict:
    """The unprotected reference: the user's own learning rate and batch size, no noise, no clipping."""
    return {"noise_multiplier": 0.0, "max_grad_norm": 1.0,
            "learning_rate": training["learning_rate"], "batch_size": training["batch_size"]}


def _search_baseline(parts: dict, out: Path, *, n_shadow: int, delta: float,  # noqa: PLR0913
                     max_physical_batch: int, device: str) -> dict:
    """Train and audit an unprotected model on the search subset, once per run.

    Every trial is a model trained on the subset, so the quality it is compared
    against must be one too: comparing against the user's model, trained on
    the whole split, would charge the trials for training on less data as well
    as for the protection. This is the one extra training (plus its audit) that
    makes "quality loss" during the search mean protection cost only. It is not
    the user's model; the confirm step measures that.
    """
    cached = read_json(out / BASELINE_FILE)
    if cached and Path(cached.get("trial_dir", "")).exists():
        return cached
    config = _baseline_config(parts["training"])
    logger.info(f"Training the unprotected reference on the search subset: {config}")
    result, mia = train_and_audit(
        parts["recipe"], parts["population"], parts["target"], out / "baseline", config,
        proxy_fpr=parts["proxy_fpr"], delta=delta, max_physical_batch=max_physical_batch,
        rmia_config={"num_shadow_models": n_shadow}, seed=parts["seed"], device=device,
    )
    record = json_safe({
        "scope": "search_subset",
        "config": config,
        "utility": result.utility,
        "train_accuracy": result.extras.get("train_accuracy"),
        "attack_tpr": result.tpr,
        "attack_tpr_ci": result.extras.get("attack_tpr_ci"),
        "n_members": result.extras.get("n_members"),
        "gated": bool(result.extras.get("gated")),
        "trial_dir": result.extras.get("trial_dir"),
    })
    (out / BASELINE_FILE).write_text(json.dumps(record, indent=2))
    return record


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

    def status(**fields: Any) -> None:  # noqa: ANN401
        write_status(out, model_name=model_name, n_trials=n_trials, proxy_fpr=parts["proxy_fpr"],
                     resolution_warning=parts["resolution_warning"],
                     n_members=len(parts["population"].train_indices),
                     model_utility=parts["model"].get("test_accuracy"), **fields)

    status(status="running")
    baseline = _search_baseline(parts, out, n_shadow=n_shadow, delta=delta,
                                max_physical_batch=max_physical_batch, device=device)
    baseline_fields = {"baseline_utility": baseline.get("utility"), "baseline_attack_tpr": baseline.get("attack_tpr"),
                       "baseline_attack_tpr_ci": baseline.get("attack_tpr_ci"), "baseline_source": baseline.get("scope")}
    status(status="running", **baseline_fields)

    def make_objective(cap: int):  # noqa: ANN202
        return make_dpsgd_objective(
            parts["recipe"], parts["population"], parts["target"], out,
            proxy_fpr=parts["proxy_fpr"], delta=delta, max_physical_batch=cap,
            rmia_config={"num_shadow_models": n_shadow}, seed=parts["seed"], device=device,
        )

    study = optimize(_with_oom_backoff(make_objective, max_physical_batch), parts["space"], out,
                     n_trials=n_trials, proxy_fpr=parts["proxy_fpr"], seed=parts["seed"])

    front = pareto_trials(study)
    status(status="done", best_indices=[t.number for t in front], **baseline_fields)
    logger.info(f"Search finished: {len(study.trials)} trials, {len(front)} on the frontier.")


def verification_path(out: Path, index: int) -> Path:
    """Where the verification of trial ``index`` is recorded."""
    return out / f"verification_{index}.json"


def verification_attack_path(out: Path, index: int) -> Path:
    """Where the RMIA result behind a verification is kept, in the shape the results view reads."""
    return out / f"verification_{index}_attack.json"


def _full_population(out: Path, dataset: Any, meta: Any, seed: int) -> Population:  # noqa: ANN401
    """The user's-size audit population: the model's own members plus as many non-members.

    Non-members are the model's own test split first (so utility is scored on
    exactly the records the user's accuracy was), then other records the model
    never trained on, until there are as many as members. Cached per model.
    """
    members = [int(i) for i in _field(meta, "train_indices")]
    test_split = [int(i) for i in _field(meta, "test_indices") or []]
    cached = read_json(out / FULL_POPULATION_META)
    if cached and Path(cached.get("path", "")).exists() and cached.get("n_members") == len(members):
        cached.pop("n_members", None)
        return Population(**cached)
    seen = set(members)
    test_split = [i for i in test_split if i not in seen]
    unseen = set(test_split)
    rest = [i for i in np.random.default_rng(seed).permutation(len(dataset)).tolist() if i not in seen and i not in unseen]
    non_members = test_split + rest[:max(0, len(members) - len(test_split))]
    population = materialize_split(dataset, out, members=members, non_members=non_members,
                                   utility=test_split or None, name=FULL_POPULATION_NAME)
    (out / FULL_POPULATION_META).write_text(json.dumps({**population.__dict__, "n_members": len(members)}))
    return population


def _attack_record(mia: Any, population: Population) -> dict:  # noqa: ANN401
    """The RMIA result in the shape the results view reads, with audit indices as dataset indices.

    The attack reports positions in the population file; every consumer of a
    result row (the records tab, the risk layer's exposed members) indexes the
    job's dataset, so the positions are mapped back through the population's
    source order. A population written before that order was recorded yields
    None, which the consumers already treat as "cannot be named".
    """
    record = _serialise_result(mia)
    record["audit_indices"] = population.to_dataset_indices(record.get("audit_indices"))
    return record


def _write_verification(out: Path, index: int, **fields: Any) -> None:  # noqa: ANN401
    verification_path(out, index).write_text(json.dumps(json_safe({"pid": os.getpid(), "index": index, **fields}),
                                                        indent=2))


def run_verification(job_dir: Path, model_name: str, index: int) -> None:
    """Measure one setting on the user's own model: retrain at full size, then audit it.

    The search ranked the setting on a subset; the number the user adopts must
    describe their model. So the setting is retrained on the clicked model's own
    training split, audited with a population of its members plus as many
    non-members (the 2:1 rule, at the user's size), and its utility is scored
    on the model's own test split and compared with the model's own accuracy.
    Quality loss is then the cost of protection alone, and the epsilon is the
    one that holds at the user's N and step count.

    When the model's split is unknown (no usable model_metadata.pkl) the
    setting's search-subset model is re-audited instead, with more shadow
    models, and the result says so: it then describes a 5,000-member model,
    not the user's.
    """
    out = pet_dir(job_dir, model_name)
    job = json.loads((job_dir / "state.json").read_text())
    request = read_json(out / REQUEST_FILE) or {}

    study = load_study(out)
    trial = next((t for t in (study.trials if study else []) if t.number == index), None)
    if trial is None or "trial_dir" not in trial.user_attrs:
        raise KeyError(f"No evaluated setting with index {index}.")
    attrs = trial.user_attrs
    estimated = {"attack_tpr": attrs.get("tpr"), "attack_tpr_ci": attrs.get("attack_tpr_ci"),
                 "utility": attrs.get("utility"), "n_members": attrs.get("n_members")}
    _write_verification(out, index, status="running", estimated=estimated)

    parts = build(job, job_dir, model_name, request)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proxy_fpr = float(study.user_attrs.get("proxy_fpr", parts["proxy_fpr"]))
    delta = float(request.get("delta") or DEFAULT_DELTA)
    max_physical_batch = int(request.get("max_physical_batch") or DEFAULT_MAX_PHYSICAL_BATCH)
    config = dict(attrs.get("config") or trial.params)
    meta = _metadata(job_dir / "models" / model_name)
    model_utility = parts["model"].get("test_accuracy")
    if model_utility is None:
        model_utility = _field(_field(meta, "test_result"), "accuracy")
    notes: list[str] = []

    if _field(meta, "train_indices"):
        population = _full_population(out, parts["dataset"], meta, parts["seed"])
        n_members, n_non = len(population.train_indices), len(population.test_indices)
        if n_non < n_members:
            notes.append(f"Only {n_non} records the model never trained on were available as non-members for "
                         f"{n_members} members, so the reference models trained on fewer points than yours.")
        warning = resolution_warning(n_non, proxy_fpr)
        if warning:
            raise ValueError(f"{n_non} non-members cannot resolve the search's {proxy_fpr:.0%} false-alarm rate.")
        logger.info(f"Retraining setting {index} on the model's own split: {n_members} members, "
                    f"{n_non} non-members, config={config}")
        result, mia = train_and_audit(
            parts["recipe"], population, parts["target"], out / "confirm" / f"trial_{index:04d}", config,
            proxy_fpr=proxy_fpr, delta=delta, max_physical_batch=max_physical_batch,
            rmia_config={"num_shadow_models": VERIFY_N_SHADOW}, seed=parts["seed"], device=device,
        )
        scope = "user_model"
        if mia is not None:
            verification_attack_path(out, index).write_text(json.dumps(json_safe(_attack_record(mia, population))))
        resolution = None
        if result.extras.get("gated"):
            resolution = "the retrained model did not learn (utility at the majority-class rate), so it was not attacked"
        elif result.extras.get("degenerate_audit"):
            resolution = "every attack score tied, so no operating point could be resolved"
        verified = {
            "scope": scope,
            "attack_tpr": result.tpr,
            "attack_tpr_table": (None if result.extras.get("degenerate_audit") or mia is None
                                 else tpr_at_fixed_fpr(mia, proxy_fpr)),
            "attack_tpr_ci": result.extras.get("attack_tpr_ci"),
            "n_members": n_members,
            "n_non_members": n_non,
            "n_nonmember_false_positives": result.extras.get("n_nonmember_false_positives"),
            "utility": result.utility,
            "train_accuracy": result.extras.get("train_accuracy"),
            "baseline_utility": model_utility,
            "epsilon": result.extras.get("epsilon"),
            "delta": delta,
            "accountant": result.extras.get("accountant"),
            "n_shadow": VERIFY_N_SHADOW,
            "proxy_fpr": proxy_fpr,
            "realized_fpr": result.extras.get("realized_fpr"),
            "resolution_warning": resolution,
            "trial_dir": result.extras.get("trial_dir"),
            "notes": notes,
        }
    else:
        notes.append("The model's own training split is unknown (no usable model_metadata.pkl), so this re-audits "
                     "the search-scale model with more reference models. It describes a model trained on "
                     f"{len(parts['population'].train_indices)} records, not yours, and quality is not compared.")
        mia = reaudit_trial(attrs["trial_dir"], population_path=parts["population"].path, target=parts["target"],
                            rmia_config={"num_shadow_models": VERIFY_N_SHADOW}, seed=parts["seed"])
        tpr, realized_fpr, degenerate = resolved_proxy_tpr(mia, proxy_fpr)
        verification_attack_path(out, index).write_text(json.dumps(json_safe(_attack_record(mia, parts["population"]))))
        verified = {
            "scope": "search_subset",
            "attack_tpr": tpr,
            "attack_tpr_table": None if degenerate else tpr_at_fixed_fpr(mia, proxy_fpr),
            **tpr_uncertainty(mia, tpr, realized_fpr),
            # The model is not retrained, so its utility and epsilon are the trial's.
            "utility": attrs.get("utility"),
            "train_accuracy": attrs.get("train_accuracy"),
            "baseline_utility": None,
            "epsilon": attrs.get("epsilon"),
            "delta": attrs.get("delta"),
            "accountant": attrs.get("accountant"),
            "n_shadow": VERIFY_N_SHADOW,
            "proxy_fpr": proxy_fpr,
            "realized_fpr": realized_fpr,
            "resolution_warning": ("every attack score tied, so no operating point could be resolved"
                                   if degenerate else None),
            "trial_dir": attrs["trial_dir"],
            "notes": notes,
        }

    _write_verification(out, index, status="done", estimated=estimated, verified=verified)
    logger.info(f"Verified setting {index} ({verified['scope']}): TPR {verified['attack_tpr']} at realized FPR "
                f"{verified['realized_fpr']} (search estimated {estimated['attack_tpr']}).")


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
            _write_verification(out, args.verify, status="failed",
                                estimated={"attack_tpr": None, "utility": None}, error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
