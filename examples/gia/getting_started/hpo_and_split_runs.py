#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tune an attack with Optuna using only what the attacker can measure, then run its stages one at a time."""
from __future__ import annotations

import dataclasses

import torch

from examples.gia.common import cli, results, runner
from leakpro.attacks.gia_attacks.modular import TargetInfo, run_attack, simulate_client
from leakpro.attacks.gia_attacks.modular.presets import inverting_gradients


def add_args(p) -> None:
    p.add_argument("--trials", type=int, default=None, help="Optuna trials (default 20, quick 3).")


def matching_loss(attack, model, reconstruction, observation, target, shape) -> float:
    """Cosine distance between the client's gradient and the gradient the reconstruction would produce.

    The attacker can compute this without the private images, so it is a legitimate tuning objective.
    """
    simulated = simulate_client(model, reconstruction, target, task=attack.task, modality=attack.modality, input_shape=shape)
    a = torch.cat([g.flatten() for g in simulated.gradients])
    b = torch.cat([g.flatten() for g in observation.gradients])
    return float(1 - torch.nn.functional.cosine_similarity(a, b, dim=0))


def main(argv: list[str] | None = None) -> dict:
    """Tune the learning rate and TV weight of Inverting Gradients, then split a two-stage run at the stage boundary."""
    import optuna  # noqa: PLC0415

    a = cli.parse(__doc__, argv, extra=add_args, plots=False)
    batch, model, device = runner.setup(a, mode="eval")
    shape, target = tuple(batch.images.shape), TargetInfo(value=batch.labels)
    base = inverting_gradients(batch.mean, batch.std, device, max_iter=runner.iters(a, 500))
    observation = simulate_client(model, batch.images, target, task=base.task, modality=base.modality, input_shape=shape)
    print("tunable surface:", sorted(base.tunables()))

    def objective(trial: optuna.Trial) -> float:
        attack = base.with_overrides({
            "0.driver.lr": trial.suggest_float("lr", 0.01, 1.0, log=True),
            "0.loss.tv.weight": trial.suggest_float("tv_weight", 1e-5, 1e-1, log=True),
        })
        state, _ = run_attack(attack, model, observation, shape, device, target=target)
        return matching_loss(attack, model, state.input, observation, target, shape)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=a.seed))
    study.optimize(objective, n_trials=a.trials or (3 if a.quick else 20))
    tuned = base.with_overrides({"0.driver.lr": study.best_params["lr"], "0.loss.tv.weight": study.best_params["tv_weight"]})

    # Two stages (the tuned search, then a fine-tuning pass at a tenth of the learning rate) run in one go and in two halves.
    refine = tuned.stages[0].with_overrides({"driver.lr": study.best_params["lr"] / 10})
    two_stage = dataclasses.replace(tuned, stages=[tuned.stages[0], refine])
    rows = {}
    full, ctx = run_attack(two_stage, model, observation, shape, device, target=target)
    first, _ = run_attack(two_stage, model, observation, shape, device, target=target, stop=1)
    resumed, _ = run_attack(two_stage, model, observation, shape, device, target=target, start=1, input_state=first)
    print("split run equals full run:", torch.allclose(full.input, resumed.input, atol=1e-6))
    for name, state in (("Default settings", run_attack(base, model, observation, shape, device, target=target)[0]),
                        ("Tuned settings", first), ("Tuned + refinement", full)):
        rows[name] = results.per_image(ctx.evaluator.evaluate(state.input, batch.images, ctx=ctx, state=state))
    return runner.finish(a, rows, best_params=study.best_params, trials=len(study.trials))


if __name__ == "__main__":
    main()
