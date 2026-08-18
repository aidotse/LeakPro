"""Optuna hyperparameter search for the DETR gradient inversion experiments.

This mirrors the protocol the coco_yolo_1_image example used to get its reconstructions: search the
total variation weight, the attack learning rate, median pooling and the top-10-norms switch, and let
optuna also choose which of several candidate client images to attack, because how much an image
leaks varies a great deal between images. Fixed hand picked hyperparameters are not a fair test of
whether an architecture leaks, since the attack learning rate alone is searched over six orders of
magnitude.

The searched objective is LPIPS by default, not SSIM: SSIM scores local luminance and contrast
structure and on these reconstructions it sits in the noise floor (~0.002) while still ranking
essentially noise, whereas LPIPS compares deep features and separates a partially recovered image
from noise much more sharply. LPIPS is a distance, so it is reported negated and optuna keeps
maximizing.
"""
import argparse
import time

import optuna
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
from leakpro.schemas import OptunaConfig
from leakpro.utils.seed import seed_everything

p = argparse.ArgumentParser()
p.add_argument("--name", default="optuna")
p.add_argument("--attack", default="inverting", choices=["inverting", "base"])
p.add_argument("--trials", type=int, default=40)
p.add_argument("--iters", type=int, default=3000, help="Iterations per trial.")
p.add_argument("--check-interval", type=int, default=500,
               help="Pruning check interval; must be a multiple of 250, which is the attack's yield period.")
p.add_argument("--num-trial-images", type=int, default=5,
               help="How many candidate client images optuna may choose between.")
p.add_argument("--metric", default="lpips", choices=["lpips", "ssim"],
               help="Reconstruction quality metric the search optimizes. LPIPS is a distance, so it is "
                    "negated and the study still maximizes; the reported best value is -LPIPS.")
add_model_args(p)
add_data_args(p)
add_loss_args(p)
args = p.parse_args()

seed_everything(1234)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(args)
model.eval().to(dev)

if args.image_id is not None and args.num_trial_images != 1:
    # One explicit image means there is nothing to choose between; search hyperparameters only.
    print(f"--image-id is set, forcing --num-trial-images 1 (was {args.num_trial_images}).", flush=True)
    args.num_trial_images = 1
if args.fixed_match and args.num_trial_images != 1:
    # The frozen assignment belongs to one specific image, so optuna must not swap the image out
    # from under it. Search hyperparameters only, on a single client.
    print(f"--fixed-match is set, forcing --num-trial-images 1 (was {args.num_trial_images}).", flush=True)
    args.num_trial_images = 1

trial_data = []
for i in range(args.num_trial_images):
    loader, data_mean, data_std = build_loader(args, start_idx=args.start_idx + i)
    trial_data.append(loader)

configs = InvertingConfig() if args.attack == "inverting" else GIABaseRunningConfig()
configs.optimizer = MetaSGD(lr=0.1)
configs.criterion = build_criterion(args, model, trial_data[0], dev)
configs.data_extension = GiaImageDetrExtension()
configs.at_iterations = args.iters
configs.similarity_metric = args.metric

if args.attack == "inverting":
    attack = InvertingGradients(model, trial_data[0], data_mean, data_std, train_fn=train_detr,
                                configs=configs, optuna_trial_data=trial_data)
else:
    attack = GIABaseRunning(model, trial_data[0], data_mean, data_std, train_fn=train_detr,
                            configs=configs, optuna_trial_data=trial_data, exp_name=args.name)

optuna_config = OptunaConfig()
optuna_config.n_trials = args.trials
optuna_config.check_interval = args.check_interval
optuna_config.direction = "maximize"
optuna_config.pruner = optuna.pruners.MedianPruner(n_warmup_steps=args.check_interval)

print(f"[{args.name}] model={model_tag(args)} attack={args.attack} metric={args.metric} "
      f"trials={args.trials} iters={args.iters} img_size={args.img_size} "
      f"images={args.num_trial_images}", flush=True)
t0 = time.time()
attack.run_with_optuna(optuna_config=optuna_config)
print(f"[{args.name}] SEARCH DONE in {time.time() - t0:.0f}s", flush=True)
