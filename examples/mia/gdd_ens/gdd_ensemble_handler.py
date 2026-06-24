#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GDD-ENS ensemble model handler — trains/evaluates a 10-MLP GddEnsemble (role="model").

Used for BOTH the target (called from gdd_ensemble_main.ipynb) and every shadow model (called by
LeakPro's ShadowModelHandler). "One shadow = one ensemble": each shadow rebuilt from the target
metadata is a full GddEnsemble, trained here on that shadow's in-split.

Training notes:
- All 10 members are trained in a single pass per epoch, each with its OWN Adam optimizer built from
  the member's Table S6 learning_rate / weight_decay. The ``optimizer`` argument LeakPro passes
  (reconstructed from the target metadata) is therefore nominal and intentionally ignored — a single
  optimizer cannot express 10 different per-member learning rates.
- No oversampling and no per-member StratifiedShuffleSplit fold (see the example README / plan):
  symmetric target/shadow training for clean LiRA/RMIA calibration, and SSS is unsafe on the rare
  classes of this 38-way imbalanced population without the paper's oversampling. Ensemble diversity
  comes from the 10 distinct architectures + random init/dropout.
"""

import torch
from torch import cuda, device, nn, no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.schemas import EvalOutput, TrainingOutput

from utils.gdd_ensemble import TABLE_S6


class GddEnsembleModelHandler(AbstractInputHandler, role="model"):
    """Train/eval handler for the GDD-ENS 10-MLP ensemble. No dataset logic."""

    def train(
        self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,  # noqa: ARG002 - nominal; per-member optimizers built below
        epochs: int = None,
    ) -> TrainingOutput:
        if epochs is None:
            raise ValueError("epochs not found in configs")
        if not hasattr(model, "members"):
            raise ValueError("GddEnsembleModelHandler expects a GddEnsemble (model.members missing)")

        dev = device("cuda" if cuda.is_available() else "cpu")
        model.to(dev)
        crit = criterion if criterion is not None else nn.CrossEntropyLoss()

        # One Adam per member, hyperparameters from Table S6 (see module docstring on why the
        # passed-in optimizer is ignored).
        optimizers = [
            optim.Adam(member.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            for member, cfg in zip(model.members, TABLE_S6)
        ]

        for epoch in range(epochs):
            for member in model.members:
                member.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for inputs, labels in pbar:
                inputs = inputs.to(dev, non_blocking=True)
                labels = labels.to(dev, non_blocking=True).long().view(-1)
                for member, opt in zip(model.members, optimizers):
                    opt.zero_grad()
                    loss = crit(member(inputs), labels)
                    loss.backward()
                    opt.step()

        train_metrics = self._evaluate(dataloader, model, crit, dev)
        model.to("cpu")
        return TrainingOutput(model=model, metrics=train_metrics)

    def eval(self, loader: DataLoader, model: nn.Module, criterion: nn.Module) -> EvalOutput:
        dev = device("cuda" if cuda.is_available() else "cpu")
        model.to(dev)
        return self._evaluate(loader, model, criterion, dev)

    @staticmethod
    def _evaluate(loader: DataLoader, model: nn.Module, criterion: nn.Module, dev: torch.device) -> EvalOutput:  # noqa: ARG004
        """Ensemble accuracy/loss from the averaged-softmax forward (model.forward = log mean prob).

        ``model(x)`` already returns log-probabilities, so loss is NLL on them directly. The passed
        ``criterion`` (CrossEntropyLoss, used to train the raw-logit members) would apply a second
        log-softmax and is intentionally not used for the ensemble metric.
        """
        model.eval()
        loss, correct, total = 0.0, 0, 0
        with no_grad():
            for data, target in loader:
                data = data.to(dev)
                target = target.to(dev).long().view(-1)
                log_probs = model(data)
                loss += nn.functional.nll_loss(log_probs, target, reduction="sum").item()
                correct += log_probs.argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        return EvalOutput(accuracy=float(correct) / total, loss=loss / total)
