"""ImageNet attack (Fig. `imagenet_architectures`) with freshly initialized model weights in every Optuna trial.

Same setup as rerun_imagenet_attack.py (torchvision model with weights=None, eval mode, GIABase with its
default config, Optuna TPE with NopPruner, fish image = Imagenette 160px train index 19, proxy index 20),
except for the model weights. rerun_imagenet_attack.py builds the model once after seed_everything(1234), and
every trial deep-copies it, so all trials attack the same weights. Here trial t rebuilds the model after
seed_everything(INIT_SEED_BASE + t), so each trial gets the default random init a new training run would
start from (and therefore its own client gradient). The seeds are recorded so imagenet_probe.py --reinit can
rebuild the same weights: probe seed s uses init seed INIT_SEED_BASE + s, i.e. the weights of trial s.

What stays fixed across trials, as in the original code: the target/proxy images and the reconstruction's
random start (the Optuna loop calls seed_everything(OptunaConfig.seed) after resetting the attack).

Run from this directory (run_reinit_attack.sh starts all five models across the available GPUs):
  python3 reinit_imagenet_attack.py --model swin_v2_t --n-trials 30
Results go to leakpro_output/results/reinit_<model>/: per-trial GIAResults (optunatrial<t>), trial_inits.jsonl
(init seed and weight fingerprint, written when each trial starts), trials.csv and optuna_results.txt.
"""
import typing

if not hasattr(typing, "Self"):  # leakpro.fl_utils.data_utils imports typing.Self, which is 3.11+
    typing.Self = typing.TypeVar("Self")

import argparse
import json
import os

import optuna
import torch
import torchvision
from _data import get_imagenette_loader
from torchvision import models

from leakpro.attacks.gia_attacks.gia_estimate import GIABase
from leakpro.attacks.utils.hyperparameter_tuning.optuna import optuna_optimal_hyperparameters
from leakpro.schemas import OptunaConfig
from leakpro.utils.seed import seed_everything

# ViT's attention has no double backward in the fused SDPA kernels on older torch; the math kernel is exact
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

MODELS = {"swin_t": models.swin_t, "swin_v2_t": models.swin_v2_t, "convnext_t": models.convnext_tiny,
          "maxvit_t": models.maxvit_t, "vit_b_16": models.vit_b_16}
TARGET_IDX = 19
PROXY_IDX = 20
INIT_SEED_BASE = 1234  # trial 0 gets the weights of the original runs (seed_everything(1234))

# _data.get_imagenette_loader passes download=True, which torchvision raises on once the data is extracted
_Imagenette = torchvision.datasets.Imagenette
torchvision.datasets.Imagenette = lambda *a, **kw: _Imagenette(
    *a, **{**kw, "download": not os.path.isdir(os.path.join(kw.get("root", "./data"), "imagenette2-160"))})


def build_model(name: str, init_seed: int) -> torch.nn.Module:
    """The torchvision model with its default random init under seed_everything(init_seed), in eval mode."""
    seed_everything(init_seed)
    model = MODELS[name](weights=None, num_classes=1000)
    return model.eval()


def weight_fingerprint(model: torch.nn.Module) -> float:
    """Sum of all parameters in float64, to check that two runs built the same weights."""
    return sum(p.detach().double().sum().item() for p in model.parameters())


class ReinitGIABase(GIABase):
    """GIABase whose Optuna trials each attack a newly initialized model."""

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__(build_model(model_name, INIT_SEED_BASE), *args, **kwargs)
        self.model_name = model_name

    def suggest_parameters(self, trial: optuna.trial.Trial) -> None:
        super().suggest_parameters(trial)
        # reset_attack, which the Optuna loop calls next, deep-copies original_model and recomputes g*
        init_seed = INIT_SEED_BASE + trial.number
        self.original_model = build_model(self.model_name, init_seed)
        fingerprint = weight_fingerprint(self.original_model)
        trial.set_user_attr("init_seed", init_seed)
        trial.set_user_attr("weight_fingerprint", fingerprint)
        with open(os.path.join(self.attack_cache_folder_path, "trial_inits.jsonl"), "a") as f:
            f.write(json.dumps({"trial": trial.number, "init_seed": init_seed, "weight_fingerprint": fingerprint,
                                "params": trial.params}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    client_loader, data_mean, data_std = get_imagenette_loader(start_idx=TARGET_IDX, num_images=1, batch_size=1,
                                                               num_workers=2)
    proxy_loader, _, _ = get_imagenette_loader(start_idx=PROXY_IDX, num_images=1, batch_size=1, num_workers=2)
    attack = ReinitGIABase(args.model, client_loader, data_mean, data_std, proxy_loader=proxy_loader,
                           exp_name=f"reinit_{args.model}")
    study = optuna_optimal_hyperparameters(attack, OptunaConfig(n_trials=args.n_trials,
                                                                pruner=optuna.pruners.NopPruner()))
    study.trials_dataframe().rename(columns={"value": "ssim"}).to_csv(
        os.path.join(attack.attack_cache_folder_path, "trials.csv"), index=False)


if __name__ == "__main__":
    main()
