#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Reconstruct a client's images from one gradient with the Inverting Gradients preset (about 20 lines of attack code)."""
from __future__ import annotations

import torch

from examples.gia.common import cli, data, models, results
from leakpro.attacks.gia_attacks.modular import TargetInfo, run_attack, simulate_client
from leakpro.attacks.gia_attacks.modular.presets import inverting_gradients


def main(argv: list[str] | None = None) -> dict:
    """Run the attack and return the per-image metrics."""
    a = cli.parse(__doc__, argv, plots=False)
    device = cli.device_for(a)
    torch.manual_seed(a.seed)
    batch = data.image_batch(a.quick, a.num_images or 4, a.data_dir, device)
    model = (models.tiny_cnn(batch.num_classes) if a.quick else models.cifar_resnet(batch.num_classes)).to(device).train()

    steps = 30 if a.quick else max(1, round(4000 * a.iter_scale))  # --iter-scale shortens or lengthens the full run
    attack = inverting_gradients(batch.mean, batch.std, device, max_iter=steps)
    target = TargetInfo(value=batch.labels)
    observation = simulate_client(model, batch.images, target, task=attack.task, modality=attack.modality,
                                  input_shape=tuple(batch.images.shape))
    state, ctx = run_attack(attack, model, observation, tuple(batch.images.shape), device, target=target)
    evaluation = ctx.evaluator.evaluate(state.input, batch.images, ctx=ctx, state=state)

    rows = {"Inverting Gradients": results.per_image(evaluation)}
    results.save(a.out, rows, {"command": a.command, "quick": a.quick, "seed": a.seed})
    results.emit(rows)
    return rows


if __name__ == "__main__":
    main()
