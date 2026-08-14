"""Shared model and data construction for the DETR gradient inversion experiments.

The attack runner, the optuna search and the diagnostics all need to build the same models and
loaders from the same flags, so the argument definitions live here rather than being duplicated.
"""
import argparse

import torch
from coco import get_coco_detr_loader
from detr_small import VARIANTS, detr_small
from model import ComputeLoss, FixedMatchComputeLoss, detr_resnet50, unfreeze_batch_norm
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add the model selection flags."""
    group = parser.add_argument_group("model")
    group.add_argument("--model", default="detr50", choices=["detr50", "small"],
                       help="detr50 is the stock transformers DETR; small is the configurable one.")
    group.add_argument("--pretrained", type=int, default=1,
                       help="detr50: load the COCO checkpoint. small: load ImageNet backbone weights.")
    group.add_argument("--unfreeze-bn", type=int, default=0,
                       help="detr50 only: convert frozen BatchNorm buffers into trainable parameters.")
    group.add_argument("--variant", default="r18", choices=list(VARIANTS),
                       help="small only: which point on the architecture ladder to build.")
    group.add_argument("--backbone", default=None, choices=["resnet18", "resnet34", "resnet50"],
                       help="small only: override the variant's backbone family.")
    group.add_argument("--stages", type=int, default=None,
                       help="small only: override the number of residual stages (output stride 2^(1+stages)).")
    group.add_argument("--enc-layers", type=int, default=None, help="small only: override encoder depth.")
    group.add_argument("--dec-layers", type=int, default=None, help="small only: override decoder depth.")
    group.add_argument("--small-stem", type=int, default=None,
                       help="small only: replace the stride 4 stem with a stride 1 one.")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """Add the client data selection flags.

    The number of annotated objects is a first class experimental factor: each ground truth object is
    one query that receives positive box and class supervision, so it sets how much of the image the
    shared gradient is actually a function of. An image with a single object constrains far less of
    the reconstruction than a crowded one.
    """
    group = parser.add_argument_group("data")
    group.add_argument("--img-size", type=int, default=256)
    group.add_argument("--min-objects", type=int, default=1,
                       help="Minimum annotated objects in the attacked image.")
    group.add_argument("--max-objects", type=int, default=3,
                       help="Maximum annotated objects in the attacked image.")
    group.add_argument("--start-idx", type=int, default=0,
                       help="Index into the sorted image ids to start searching from.")
    group.add_argument("--split", default="val2017", choices=["val2017", "train2017"],
                       help="Which COCO split to attack. The YOLO example's indices refer to train2017.")
    group.add_argument("--image-id", type=int, default=None,
                       help="Attack this exact COCO image id, ignoring --start-idx and the object count "
                            "filter. Use it to reproduce a specific published image.")
    group.add_argument("--letterbox", type=int, default=1,
                       help="Fit a non square image to --img-size preserving its aspect ratio and pad with "
                            "black, matching the YOLO example pixel for pixel. 0 stretches it to the square "
                            "instead, which is what produced the result tables in CLAUDE.md.")


def add_loss_args(parser: argparse.ArgumentParser) -> None:
    """Add the client criterion flags."""
    group = parser.add_argument_group("loss")
    group.add_argument("--fixed-match", type=int, default=0,
                       help="Freeze the Hungarian assignment to the one the true image induces. Leaves the "
                            "client's shared update bit-identical but makes the attacker's objective smooth.")
    group.add_argument("--refresh-every", type=int, default=0,
                       help="With --fixed-match, re-solve the matching every N loss calls instead of never. "
                            "0 never re-solves, which is the oracle upper bound.")


def build_model(args: argparse.Namespace) -> Module:
    """Build the model described by the parsed arguments."""
    if args.model == "detr50":
        model = detr_resnet50(pretrained=bool(args.pretrained))
        if args.unfreeze_bn:
            model = unfreeze_batch_norm(model)
        return model

    overrides = {"pretrained_backbone": bool(args.pretrained)}
    if args.backbone is not None:
        overrides["backbone"] = args.backbone
    if args.stages is not None:
        overrides["backbone_stages"] = args.stages
    if args.enc_layers is not None:
        overrides["num_encoder_layers"] = args.enc_layers
    if args.dec_layers is not None:
        overrides["num_decoder_layers"] = args.dec_layers
    if args.small_stem is not None:
        overrides["small_stem"] = bool(args.small_stem)
    return detr_small(args.variant, **overrides)


def build_loader(args: argparse.Namespace, start_idx: int = None) -> tuple[DataLoader, Tensor, Tensor]:
    """Build a single image client loader from the parsed data arguments."""
    image_id = getattr(args, "image_id", None)
    return get_coco_detr_loader(
        num_images=1, img_size=args.img_size, batch_size=1,
        start_idx=args.start_idx if start_idx is None else start_idx,
        split=getattr(args, "split", "val2017"),
        min_objects=args.min_objects, max_objects=args.max_objects,
        letterbox=bool(getattr(args, "letterbox", 1)),
        image_ids=None if image_id is None else [image_id])


def build_criterion(args: argparse.Namespace, model: Module, client_loader: DataLoader,
                    device: torch.device) -> Module:
    """Build the client criterion, freezing the bipartite matching if asked to.

    The frozen assignment is read off the real client image, which is the only assignment that leaves
    the honest client's update unchanged, so the reconstruction target stays exactly where it was.
    """
    if not getattr(args, "fixed_match", 0):
        return ComputeLoss(model)
    criterion = FixedMatchComputeLoss(model, refresh_every=getattr(args, "refresh_every", 0))
    image, target = client_loader.dataset[0]
    criterion.set_assignment_from(image.unsqueeze(0).to(device),
                                  [{k: v.to(device) for k, v in target.items()}])
    return criterion


def model_tag(args: argparse.Namespace) -> str:
    """Short human readable description of the built model, for logging."""
    if args.model == "detr50":
        return f"detr50(pretrained={bool(args.pretrained)},unfrozen_bn={bool(args.unfreeze_bn)})"
    return f"small:{args.variant}(pretrained_backbone={bool(args.pretrained)})"
