"""To be used in the future maybe."""
from collections import defaultdict
from typing import Self

import torch
from torch import nn
from torch.utils.data import DataLoader

from leakpro.utils.logger import logger


def pre_train(model: nn.Module, trainloader: DataLoader, epochs: int = 10) -> None:
    """Pre train a model for a specified amount of epochs."""
    loss_fn = Classification()
    {"dtype": torch.float, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    stats = defaultdict(list)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                    weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[120 // 2.667, 120 // 1.6,
                                                                     120 // 1.142], gamma=0.1)

    for _ in range(epochs):
        logger.info(stats)
        model.train()
        setup = {"dtype": torch.float, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
        model.to(setup["device"])
        epoch_loss, epoch_metric = 0, 0
        for _, (inputs, targets) in enumerate(trainloader):
            optimizer.zero_grad()
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup["device"], non_blocking=False)
            outputs = model(inputs)
            loss, _, _ = loss_fn(outputs, targets)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            metric, name, _ = loss_fn.metric(outputs, targets)
            epoch_metric += metric.item()

        scheduler.step()

        stats["train_losses"].append(epoch_loss / (len(trainloader) + 1))
        stats["train_" + name].append(epoch_metric / (len(trainloader) + 1))
        logger.info(stats)


class Classification():
    """A classical NLL loss for classification. Evaluation has the softmax baked in.

    The minimized criterion is cross entropy, the actual metric is total accuracy.
    """

    def __init__(self: Self) -> None:
        """Init with torch MSE."""
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                 reduce=None, reduction="mean")

    def __call__(self, x=None, y=None):  # noqa: ANN001, ANN101, ANN204
        """Return l(x, y)."""
        name = "CrossEntropy"
        format = "1.5f"
        if x is None:
            return name, format
        value = self.loss_fn(x, y)
        return value, name, format

    def metric(self, x=None, y=None):  # noqa: ANN001, ANN101, ANN201
        """The actually sought metric."""
        name = "Accuracy"
        format = "6.2%"
        if x is None:
            return name, format
        value = (x.data.argmax(dim=1) == y).sum().float() / y.shape[0]
        return value.detach(), name, format
