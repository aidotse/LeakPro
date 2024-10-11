"""To be used in the future maybe."""
from collections import defaultdict

import torch

# This will maybe be used in the future for experiments with larger batch sizes.

def pre_train(model, trainloader, valloader):  # noqa: ANN001, ANN201, D103
    loss_fn = Classification()
    train(model, loss_fn, trainloader, valloader)

def train(model, loss_fn, trainloader, validloader):  # noqa: ANN001, ANN201, ARG001
    """Run the main interface. Train a network with specifications from the Strategy object."""
    {"dtype": torch.float, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    stats = defaultdict(list)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                    weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[120 // 2.667, 120 // 1.6,
                                                                     120 // 1.142], gamma=0.1)

    for _epoch in range(10):
        print(stats)  # noqa: T201
        model.train()
        step(model, loss_fn, trainloader, optimizer, scheduler, stats)
        print(stats)  # noqa: T201

    return stats

def step(model, loss_fn, dataloader, optimizer, scheduler, stats):  # noqa: ANN001, ANN201
    """Step through one epoch."""
    setup = {"dtype": torch.float, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    model.to(setup["device"])
    epoch_loss, epoch_metric = 0, 0
    for batch, (inputs, targets) in enumerate(dataloader):  # noqa: B007
        # Prep Mini-Batch
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**setup)
        targets = targets.to(device=setup["device"], non_blocking=False)

        # Get loss
        outputs = model(inputs)
        loss, _, _ = loss_fn(outputs, targets)


        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        metric, name, _ = loss_fn.metric(outputs, targets)
        epoch_metric += metric.item()

    scheduler.step()

    stats["train_losses"].append(epoch_loss / (batch + 1))
    stats["train_" + name].append(epoch_metric / (batch + 1))


class Classification():
    """A classical NLL loss for classification. Evaluation has the softmax baked in.

    The minimized criterion is cross entropy, the actual metric is total accuracy.
    """

    def __init__(self) -> None:
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
        name = 'Accuracy'
        format = '6.2%'
        if x is None:
            return name, format
        value = (x.data.argmax(dim=1) == y).sum().float() / y.shape[0]
        return value.detach(), name, format
