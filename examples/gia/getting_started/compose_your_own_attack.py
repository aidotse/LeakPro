#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Build an attack from parts: cosine gradient matching, total variation and a custom loss term of your own."""
from __future__ import annotations

import torch

from examples.gia.common import cli, results, runner
from leakpro.attacks.gia_attacks.modular import Attack, FedSGD, OptimStage, Task
from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, TVLoss
from leakpro.attacks.gia_attacks.modular.components.modalities import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.representations import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.core import LossTerm


class MidGrayPrior(LossTerm):
    """A custom prior: natural photographs average to mid-gray, so penalise each image's distance from 0.5."""

    def compute(self, inputs) -> torch.Tensor:  # noqa: ANN001
        """Mean squared distance of every image's mean pixel from 0.5."""
        return (inputs.recon.mean(dim=(-3, -2, -1)) - 0.5).pow(2).mean()


def main(argv: list[str] | None = None) -> dict:
    """Run the composed attack with and without the custom term."""
    a = cli.parse(__doc__, argv, plots=False)
    batch, model, device = runner.setup(a, mode="eval")
    rows = {}
    for name, prior_weight in (("Cosine + TV", 0.0), ("Cosine + TV + mid-gray prior", 1e-2)):
        losses = {"gradient": (CosineGradientLoss(), 1.0), "tv": (TVLoss(), 1e-2)}
        if prior_weight:
            losses["mid_gray"] = (MidGrayPrior(), prior_weight)
        attack = Attack(
            modality=ImageNormalizationModality(mean=batch.mean, std=batch.std, device=device),
            task=Task(CrossEntropy(), FedSGD(), model_mode="eval"),
            evaluator=ImageEvaluator(),
            stages=[OptimStage(
                losses=losses,
                rep=PixelRepresentation(min_val=0.0, max_val=1.0),
                driver=GradientDriver(lr=0.1, max_iter=runner.iters(a, 500)),
                target_source=FixedTarget(),
                aggregator=BestLossAggregator(),
            )],
        )
        evaluation, _, _ = runner.run(attack, model, batch, device, seed=a.seed)
        rows[name] = results.per_image(evaluation)
        print(f"{name}: PSNR {evaluation.metrics['psnr']:.2f} dB")
    return runner.finish(a, rows)


if __name__ == "__main__":
    main()
