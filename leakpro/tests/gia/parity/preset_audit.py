#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Side-by-side v1 preset defaults (from the recorded fixtures) and framework preset defaults, as markdown.

    python -m leakpro.tests.gia.parity.preset_audit > preset-audit.md

The v1 column is each recorded preset's configuration with its default arguments; the framework
column is read off the Attack the framework preset builds with its default arguments.
"""
from __future__ import annotations

from typing import Any

from leakpro.attacks.gia_attacks.modular import presets
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import GaussianNoise, SignGradient
from leakpro.attacks.gia_attacks.modular.components.initializers import Uniform
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg
from leakpro.tests.gia import helpers
from leakpro.tests.gia.parity import fixtures

# v1 loss type -> concept row
V1_LOSS = {"gradient_matching": "gradient matching", "tv": "TV prior", "bn_stats": "BN-statistics prior",
           "l2": "image L2 prior", "group_consistency": "group-consistency prior"}
# framework loss term class -> concept row
FRAMEWORK_LOSS = {"CosineGradientLoss": "gradient matching", "L2GradientLoss": "gradient matching",
                  "TVLoss": "TV prior", "BNStatsLoss": "BN-statistics prior", "L2ImageNormLoss": "image L2 prior",
                  "GroupConsistencyLoss": "group-consistency prior", "EpochOrderInvariantPrior": "epoch-order prior"}

# v1 preset name -> framework preset factory
FRAMEWORK_PRESET = {
    "inverting_gradients": lambda: presets.inverting_gradients(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE),
    "idlg": lambda: presets.idlg(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE),
    "huang": lambda: presets.huang(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE),
    "gia_running": lambda: presets.gia_running(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE),
    "see_through_gradients": lambda: presets.yin(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE),
    "dimitrov": lambda: presets.dimitrov(helpers.CIFAR_MEAN, helpers.CIFAR_STD, helpers.DEVICE),
}


def _v1_row(config: dict[str, Any]) -> dict[str, Any]:
    stage, ts = config["stages"][0], config["training_simulator"]
    row = {
        "optimizer": stage["optimizer_type"]["type"].removeprefix("optimizer.") if stage["optimizer_type"] else "adam",
        "lr": stage["learning_rate"],
        "max_iter": stage["max_iterations"],
        "scheduler": stage["scheduler_type"]["type"].removeprefix("scheduler.") if stage["scheduler_type"] else None,
        "gradient sign": stage["use_gradient_sign"],
        "gradient noise std": stage["gradient_noise_std"],
        "return best": stage["return_best"],
        "early-stop patience": stage["patience"],
        "optimization space": "normalized (z)",
        "initialization": config["initialization"]["type"].removeprefix("init."),
        "box constraint": stage["constraint"]["type"].removeprefix("constraint.") if stage["constraint"] else None,
        "seeds": config["num_seeds_per_image"],
        "labels": config["label_inference"]["type"].removeprefix("label_inference."),
        "client compute mode": ts["compute_mode"],
        "client model mode": ts["model_mode"],
        "client epochs": ts["epochs"],
        "client batch size": ts["batch_size"],
        "threat model": config["threat_model_type"],
    }
    for spec in stage["losses"]:
        params = dict(spec["params"])
        weight = params.pop("weight")
        detail = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        row[f"loss: {V1_LOSS[spec['type'].removeprefix('loss.')]}"] = f"{weight}" + (f" ({detail})" if detail else "")
    if config.get("fedavg"):
        row["loss: epoch-order prior"] = f"{config['fedavg']['lambda_inv']} ({config['fedavg']['order_invariant_fn']})"
    return row


def _initializer(rep: Any) -> str:  # noqa: ANN401
    """The representation's initializer, or the default draw of its box when it names none."""
    init = rep.init if rep.init is not None else Uniform(rep.min_val, rep.max_val)
    detail = ", ".join(f"{k}={v}" for k, v in vars(init).items() if k != "space")
    return f"{type(init).__name__.lower()}({detail})"


def _threat_model(attack: Attack) -> str:
    if attack.threat_model is None:
        return "undeclared"
    return ", ".join(attack.threat_model.granted()) or "gradients only"


def _framework_row(attack: Attack) -> dict[str, Any]:
    stage = attack.stages[0]
    driver, task = stage.driver, attack.task
    fedavg = isinstance(task.protocol, FedAvg)
    row = {
        "optimizer": driver.optimizer_cls.__name__.lower(),
        "lr": driver.lr,
        "max_iter": driver.max_iter,
        "scheduler": type(driver.schedule).__name__ if driver.schedule else None,
        "gradient sign": any(isinstance(t, SignGradient) for t in driver.grad_transforms),
        "gradient noise std": next((t.std for t in driver.grad_transforms if isinstance(t, GaussianNoise)), 0.0),
        "return best": driver.return_best,
        "early-stop patience": driver.patience,
        "optimization space": "normalized (z)" if stage.rep.space == "model_input" else "data (x)",
        "initialization": _initializer(stage.rep),
        "box constraint": f"clip [{stage.rep.min_val}, {stage.rep.max_val}]" if stage.rep.clip else None,
        "seeds": attack.axes.get_size("seed") if attack.axes.get_dim("seed") is not None else 1,
        "labels": f"{type(attack.target_inference).__name__ if attack.target_inference else 'oracle'} -> {type(stage.target_source).__name__}",
        "client compute mode": "updates" if fedavg else "gradients",
        "client model mode": task.model_mode,
        "client epochs": task.protocol.local_epochs if fedavg else 1,
        "client batch size": task.protocol.batch_size if fedavg else None,
        "threat model": _threat_model(attack),
    }
    for term, weight in stage.losses.values():
        detail = type(term).__name__ + (f", strategy={term.strategy}" if hasattr(term, "strategy") else "")
        row[f"loss: {FRAMEWORK_LOSS[type(term).__name__]}"] = f"{weight} ({detail})"
    return row


def render() -> str:
    """The audit as markdown: one table per recorded v1 preset."""
    out = ["# Preset audit: v1 effective defaults vs framework defaults\n"]
    seen: set[str] = set()
    for case in fixtures.cases("loss_points") + fixtures.cases("trajectory"):
        settings = fixtures.manifest()["cases"][case]["settings"]
        preset = settings["preset"]
        if preset in seen:
            continue
        seen.add(preset)
        v1 = _v1_row(settings["preset_defaults"])
        ours = _framework_row(FRAMEWORK_PRESET[preset]())
        out.append(f"## {preset}\n\n| setting | v1 | framework |\n|---|---|---|")
        out.extend(f"| {key} | {v1.get(key, '—')} | {ours.get(key, '—')} |" for key in list(v1) + [k for k in ours if k not in v1])
        out.append("")
    return "\n".join(out)


def main() -> None:
    print(render())  # noqa: T201


if __name__ == "__main__":
    main()
