"""DETR object detector and its loss, wired up for gradient inversion attacks."""
import torch
from torch import Tensor
from torch.nn import BatchNorm2d, Module
from transformers import DetrConfig, DetrForObjectDetection
from transformers.loss.loss_for_object_detection import (
    ForObjectDetectionLoss,
    HungarianMatcher,
    ImageLoss,
)
from transformers.models.detr.modeling_detr import DetrFrozenBatchNorm2d


def detr_resnet50(pretrained: bool = True, checkpoint: str = "facebook/detr-resnet-50") -> DetrForObjectDetection:
    """Get a DETR model with a ResNet-50 backbone.

    With pretrained=False the exact same architecture is built from the checkpoint config but
    with randomly initialized weights, which is the setting of an early federated training round.
    """
    if pretrained:
        return DetrForObjectDetection.from_pretrained(checkpoint)
    return DetrForObjectDetection(DetrConfig.from_pretrained(checkpoint))


def unfreeze_batch_norm(module: Module) -> Module:
    """Replace DETR's frozen BatchNorm layers with equivalent trainable nn.BatchNorm2d layers.

    DETR stores its backbone BatchNorm scale, shift and statistics as buffers, because the original
    training recipe freezes them. In eval mode a frozen layer computes exactly the same function as an
    nn.BatchNorm2d, so this conversion leaves the detector's predictions unchanged, but it turns the
    scale and shift back into parameters. That matters for gradient inversion in two ways: their
    gradients (per channel sums over the normalized activations) become part of the shared update, and
    LeakPro's BatchNorm based regularizers, which look for nn.BatchNorm2d modules, become active.

    This models a client that fine tunes DETR with its normalization layers trainable, rather than one
    that follows the original frozen recipe.
    """
    for name, child in module.named_children():
        if isinstance(child, DetrFrozenBatchNorm2d):
            num_features = child.weight.shape[0]
            batch_norm = BatchNorm2d(num_features, eps=1e-5)
            batch_norm.weight.data.copy_(child.weight.data)
            batch_norm.bias.data.copy_(child.bias.data)
            batch_norm.running_mean.data.copy_(child.running_mean.data)
            batch_norm.running_var.data.copy_(child.running_var.data)
            setattr(module, name, batch_norm)
        else:
            unfreeze_batch_norm(child)
    return module


class ComputeLoss(Module):
    """DETR set prediction loss (Hungarian matching + classification, L1 and GIoU box loss).

    Wraps the transformers implementation so it can be used as the client criterion in GIA,
    where it is called as criterion(model_outputs, targets) and has to return a scalar which is
    differentiable all the way back to the input pixels.
    """

    def __init__(self, model: DetrForObjectDetection) -> None:
        super().__init__()
        self.config = model.config

    def forward(self, outputs: object, targets: list) -> Tensor:
        """Compute the weighted sum of the DETR losses."""
        loss, _, _ = ForObjectDetectionLoss(
            logits=outputs.logits,
            labels=targets,
            device=outputs.logits.device,
            pred_boxes=outputs.pred_boxes,
            config=self.config,
        )
        return loss


class _FixedMatcher:
    """Stands in for HungarianMatcher, returning a stored assignment instead of solving for one."""

    def __init__(self, indices: list) -> None:
        self.indices = indices

    def __call__(self, outputs: dict, targets: list) -> list:  # noqa: ARG002
        """Return the stored assignment, ignoring the current predictions."""
        return self.indices


class FixedMatchComputeLoss(Module):
    """DETR set prediction loss with the query-to-object assignment held fixed.

    DETR re-solves Hungarian matching on every forward pass. During a gradient inversion attack the
    pixels move, so the predictions move, so the assignment flips: measured on a 20 object COCO image,
    60% of object-steps get reassigned along a straight path from noise to the true image, and 9 of 20
    objects are still mismatched at 95% of the way there. Each distinct assignment defines a different
    loss and therefore a different gradient for the attack to chase, which makes the reconstruction
    objective piecewise rather than smooth.

    Freezing the assignment to the one the client's own image induces removes that. It costs nothing in
    fidelity: at the true image the frozen assignment *is* what Hungarian matching returns, so the loss
    and its gradient are identical there, and an honest client's shared update is unchanged. Only the
    attacker's path through pixel space becomes smooth. Knowing that assignment is oracle knowledge, so
    this measures whether matching instability is what blocks the attack, rather than being a
    deployable attack itself; `refresh_every` gives the practical version, which re-solves the matching
    every so many calls instead of every call.
    """

    def __init__(self, model: DetrForObjectDetection, refresh_every: int = 0) -> None:
        super().__init__()
        self.config = model.config
        self.model = model
        self.refresh_every = refresh_every
        self.indices = None
        self.calls = 0
        self.matcher = HungarianMatcher(class_cost=self.config.class_cost,
                                       bbox_cost=self.config.bbox_cost,
                                       giou_cost=self.config.giou_cost)

    def set_assignment_from(self, images: Tensor, targets: list) -> list:
        """Store the assignment the given images induce, and return it."""
        with torch.no_grad():
            outputs = self.model(images)
            indices = self.matcher({"logits": outputs.logits, "pred_boxes": outputs.pred_boxes}, targets)
        self.indices = [(query.clone(), obj.clone()) for query, obj in indices]
        return self.indices

    def forward(self, outputs: object, targets: list) -> Tensor:
        """Compute the weighted sum of the DETR losses under the frozen assignment."""
        if self.indices is None:
            raise RuntimeError("Call set_assignment_from(images, targets) before using this loss.")
        self.calls += 1
        if self.refresh_every and self.calls % self.refresh_every == 0:
            with torch.no_grad():
                self.indices = [(q.clone(), o.clone()) for q, o in self.matcher(
                    {"logits": outputs.logits.detach(), "pred_boxes": outputs.pred_boxes.detach()}, targets)]
        criterion = ImageLoss(matcher=_FixedMatcher(self.indices), num_classes=self.config.num_labels,
                              eos_coef=self.config.eos_coefficient,
                              losses=["labels", "boxes", "cardinality"]).to(outputs.logits.device)
        loss_dict = criterion({"logits": outputs.logits, "pred_boxes": outputs.pred_boxes}, targets)
        weights = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient,
                   "loss_giou": self.config.giou_loss_coefficient}
        return sum(loss_dict[k] * weights[k] for k in loss_dict if k in weights)
