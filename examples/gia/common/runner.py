#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Run one attack against a simulated client and score it: the steps every image example repeats."""
from __future__ import annotations

import torch

from examples.gia.common import cli, data, models
from leakpro.attacks.gia_attacks.modular import Attack, RunMetadata, TargetInfo, run_attack, simulate_client


def setup(args, *, num_images: int = 4, dataset: str = "cifar10", mode: str = "train",
          fixed_num_images: bool = False) -> tuple[data.ImageBatch, torch.nn.Module, torch.device]:
    """Device, private batch and victim model for an example (tiny CNN with ``--quick``, PreAct-ResNet-18 otherwise).

    ``--num-images`` overrides ``num_images`` unless ``fixed_num_images`` is set (a sweep over the batch size).
    """
    device = cli.device_for(args)
    torch.manual_seed(args.seed)
    n = num_images if fixed_num_images else args.num_images or num_images
    batch = data.image_batch(args.quick, n, args.data_dir, device, dataset=dataset)
    model = models.tiny_cnn(batch.num_classes) if args.quick else models.cifar_resnet(batch.num_classes)
    model = model.to(device)
    return batch, model.train() if mode == "train" else model.eval(), device


def iters(args, full: int, quick: int = 20) -> int:
    """Optimisation steps: ``full`` (times ``--iter-scale``) normally, ``quick`` with ``--quick``."""
    return quick if args.quick else max(1, round(full * getattr(args, "iter_scale", 1.0)))


def run(
    attack: Attack,
    model: torch.nn.Module,
    batch: data.ImageBatch,
    device: torch.device,
    *,
    oracle: bool = True,
    seed: int = 1234,
    images: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    proxy_dataloader=None,  # noqa: ANN001
):
    """Simulate the client training on ``images`` (default the whole batch), attack it and score the reconstruction.

    ``oracle`` hands the true labels to the attack; without it the attack must infer them (iDLG, joint labels).
    Returns ``(evaluation, state, ctx)``; ``evaluation.extras["psnr_per_image"]`` etc. hold the per-image metrics.
    """
    images = batch.images if images is None else images
    labels = batch.labels if labels is None else labels
    target = TargetInfo(value=labels)
    shape = tuple(images.shape)
    observation = simulate_client(model, images, target, task=attack.task, modality=attack.modality, input_shape=shape)
    state, ctx = run_attack(attack, model, observation, shape, device, target=target if oracle else None,
                            metadata=RunMetadata(seed=seed), proxy_dataloader=proxy_dataloader)
    evaluation = ctx.evaluator.evaluate(state.input, images, ctx=ctx, state=state)
    return evaluation, state, ctx


def proxy_loader(args, batch: data.ImageBatch, device: torch.device, n: int = 64, batch_size: int = 16):  # noqa: ANN201
    """Attacker-side auxiliary data in model-input space: images disjoint from the private batch, in minibatches."""
    from examples.gia.common import data as _data  # noqa: PLC0415

    aux = _data.image_batch(args.quick, n, args.data_dir, device, start_idx=10_000)
    mean = torch.tensor(batch.mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(batch.std, device=device).view(1, -1, 1, 1)
    x = (aux.images - mean) / std
    return [(x[i:i + batch_size], aux.labels[i:i + batch_size]) for i in range(0, n, batch_size)]


def finish(args, rows: dict, **config) -> dict:
    """Save ``results.json`` (with the command and settings) and print the ``RESULT`` line."""
    from examples.gia.common import results  # noqa: PLC0415

    results.save(args.out, rows, {"command": args.command, "quick": args.quick, "seed": args.seed, **config})
    results.emit(rows)
    return rows


def save_grid(out, originals: torch.Tensor, recons: dict[str, torch.Tensor], labels: torch.Tensor, rows: dict) -> None:  # noqa: ANN001
    """Plot the originals against every attack's reconstruction and keep the tensors so ``--replot`` can redraw."""
    from pathlib import Path  # noqa: PLC0415

    from examples.gia.common import viz  # noqa: PLC0415

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"originals": originals.cpu(), "recons": {k: v.cpu() for k, v in recons.items()}, "labels": labels.cpu()},
               out / "reconstructions.pt")
    replot_grid(out, {k: f"PSNR {sum(r['psnr']) / len(r['psnr']):.1f} dB" for k, r in rows.items() if "psnr" in r})


def replot_grid(out, captions: dict[str, str] | None = None) -> None:  # noqa: ANN001
    """Redraw ``grid.png`` from ``reconstructions.pt`` and (without ``captions``) the PSNRs in ``results.json``."""
    from pathlib import Path  # noqa: PLC0415

    from examples.gia.common import results, viz  # noqa: PLC0415

    out = Path(out)
    saved = torch.load(out / "reconstructions.pt", weights_only=True)
    if captions is None:
        rows = results.load(out)["rows"]
        captions = {k: f"PSNR {r['psnr']['mean']:.1f} dB" for k, r in rows.items() if "psnr" in r}
    viz.visualize_multiple_attacks(saved["originals"], saved["recons"], saved["labels"], out / "grid.png", captions=captions)
