"""Where does the DETR client update actually live, and which parts of it depend on the image?

For a client gradient from the true image, and gradients from (a) random noise and (b) a second
different real image, report per parameter group: share of total gradient norm, and cosine similarity
to the client gradient. Groups that are image dependent should show low cosine, and only those can
drive the reconstruction.
"""
import argparse
from collections import OrderedDict

import torch
from build import add_data_args, add_model_args, build_loader, build_model, model_tag
from model import ComputeLoss
from torch.utils.data import DataLoader

from leakpro.fl_utils.data_utils import CustomDetrTensorDataset, GiaImageDetrExtension, detr_collate_fn
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train_detr
from leakpro.utils.seed import seed_everything

p = argparse.ArgumentParser()
add_model_args(p)
add_data_args(p)
args = p.parse_args()

seed_everything(1234)
dev = torch.device("cuda")
model = build_model(args).eval().to(dev)
print(f"model={model_tag(args)} img_size={args.img_size} objects={args.min_objects}-{args.max_objects}")
criterion = ComputeLoss(model)
names = [n for n, _ in model.named_parameters()]
bn_params = {n for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)
             for n in (f"{n}.weight", f"{n}.bias")}


def group_of(name: str) -> str:
    """Coarse parameter group for a DETR parameter name."""
    if name in bn_params:
        return "backbone_bn"
    if "backbone" in name:
        return "backbone"
    if "input_projection" in name:
        return "input_proj"
    if "query_position_embeddings" in name:
        return "queries"
    if "model.encoder" in name:
        return "encoder"
    if "model.decoder" in name:
        return "decoder"
    if "class_labels_classifier" in name:
        return "cls_head"
    if "bbox_predictor" in name:
        return "box_head"
    return "other"


def grads_for(loader: torch.utils.data.DataLoader) -> list:
    """Client update for a loader."""
    return [g.detach() for g in train_detr(model, loader, MetaSGD(lr=0.1), criterion, 1)]


client_loader, _, _ = build_loader(args)
ext = GiaImageDetrExtension()
org_loader, original, reconstruction, labels, rec_loader = ext.get_at_data(client_loader)
print(f"attacked image objects={labels[0]['class_labels'].tolist()}")

# a second, different real image but paired with the *same* targets, to isolate image dependence
other_loader, _, _ = build_loader(args, start_idx=args.start_idx + 200)
other_img = other_loader.dataset[0][0]
other_same_labels = DataLoader(CustomDetrTensorDataset(other_img.unsqueeze(0), labels),
                               batch_size=1, shuffle=False, collate_fn=detr_collate_fn)

g_client = grads_for(org_loader)
g_noise = grads_for(rec_loader)
g_other = grads_for(other_same_labels)


def cos(a: list, b: list) -> float:
    """Global cosine over a list of tensors."""
    ca = torch.cat([t.flatten() for t in a]).double()
    cb = torch.cat([t.flatten() for t in b]).double()
    return float(torch.nn.functional.cosine_similarity(ca[None], cb[None]))


total = torch.cat([t.flatten() for t in g_client]).norm().item()
print(f"{'group':<12}{'#par':>6}{'norm share':>12}{'cos(noise)':>12}{'cos(other)':>12}")
groups = OrderedDict()
for name, gc, gn, go in zip(names, g_client, g_noise, g_other):
    groups.setdefault(group_of(name), []).append((gc, gn, go))
for gname, entries in groups.items():
    gc = [e[0] for e in entries]
    share = torch.cat([t.flatten() for t in gc]).norm().item() / total
    print(f"{gname:<12}{len(entries):>6}{share:>12.4f}{cos(gc, [e[1] for e in entries]):>12.4f}"
          f"{cos(gc, [e[2] for e in entries]):>12.4f}")
print(f"{'ALL':<12}{len(names):>6}{1.0:>12.4f}{cos(g_client, g_noise):>12.4f}{cos(g_client, g_other):>12.4f}")
