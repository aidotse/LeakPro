"""Configurable DETR GIA experiment runner."""
import argparse
import time

import torch
from build import (
    add_data_args,
    add_loss_args,
    add_model_args,
    build_criterion,
    build_loader,
    build_model,
    model_tag,
)

from leakpro.attacks.gia_attacks.gia_running import GIABaseRunning, GIABaseRunningConfig
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.data_utils import GiaImageDetrExtension
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train_detr
from leakpro.run import run_gia_attack
from leakpro.utils.seed import seed_everything

p = argparse.ArgumentParser()
p.add_argument("--name", default="probe")
p.add_argument("--iters", type=int, default=500)
p.add_argument("--tv", type=float, default=1e-2)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--median-pooling", type=int, default=1)
p.add_argument("--save", type=int, default=0)
p.add_argument("--attack", default="inverting", choices=["inverting", "base"])
p.add_argument("--bn-reg", type=float, default=None)
add_model_args(p)
add_data_args(p)
add_loss_args(p)
args = p.parse_args()

seed_everything(1234)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(args)
model.eval().to(dev)

client_loader, data_mean, data_std = build_loader(args)
_, target = client_loader.dataset[0]

configs = InvertingConfig() if args.attack == "inverting" else GIABaseRunningConfig()
configs.optimizer = MetaSGD(lr=0.1)
configs.criterion = build_criterion(args, model, client_loader, dev)
configs.data_extension = GiaImageDetrExtension()
configs.at_iterations = args.iters
configs.tv_reg = args.tv
configs.attack_lr = args.lr
configs.median_pooling = bool(args.median_pooling)
if args.bn_reg is not None:
    configs.bn_reg = args.bn_reg

if args.attack == "inverting":
    attack = InvertingGradients(model, client_loader, data_mean, data_std, train_fn=train_detr, configs=configs)
else:
    attack = GIABaseRunning(model, client_loader, data_mean, data_std, train_fn=train_detr, configs=configs,
                            exp_name=args.name)
print(f"[{args.name}] model={model_tag(args)} objects={target['class_labels'].tolist()} cfg={vars(args)}", flush=True)
t0 = time.time()
if args.save:
    res = run_gia_attack(attack, experiment_name=args.name)
else:
    attack.prepare_attack()
    res = None
    for i, sim, r in attack.run_attack():
        print(f"[{args.name}] iter={i} ssim={float(sim):.4f} best_loss={float(attack.best_loss):.4f} "
              f"t={time.time() - t0:.0f}s", flush=True)
        res = r if r is not None else res
print(f"[{args.name}] DONE ssim={float(res.SSIM_score):.4f} psnr={float(res.PSNR_score):.2f} "
      f"best_loss={float(attack.best_loss):.4f} time={time.time() - t0:.0f}s", flush=True)
