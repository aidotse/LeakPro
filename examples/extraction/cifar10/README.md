# CIFAR-10 training and sampling

Run `main.ipynb` after installing the [example dependencies](../README.md).
Device `auto` uses CUDA when available, otherwise CPU. On a Mac this backend uses CPU: the
pinned upstream diffusion code transfers float64 coefficients, which MPS does not support.
`train_config.yaml` selects the target profile; `audit.yaml` and `audit_demonstration.yaml` select attacks.
The notebook first loads and audits the official released checkpoint, then independently trains
from scratch and audits that target. Both stages also support a Carlini-only attack list.
Stage outputs and manifests are separated into `released/` and `from_scratch/` directories.

The `demonstration` target uses the architecture and hybrid loss from the [official Improved DDPM CIFAR-10 recipe](https://github.com/openai/improved-diffusion/tree/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5#models-and-hyperparameters):

| Setting | Demonstration |
| --- | --- |
| U-Net | 128 base channels, multipliers `(1, 2, 2, 2)`, 3 residual blocks per level |
| Attention | 16×16 and 8×8, 4 heads |
| Dropout | 0.3 |
| Diffusion | 4,000 steps, cosine schedule, learned variance |
| Loss | Official hybrid noise-prediction and variational loss |
| Optimizer | AdamW, learning rate 0.0001, weight decay 0 |
| Effective batch | 128, accumulated in microbatches of 32 |
| Sampling weights | EMA, decay 0.999 for the shorter experiment |
| Sampling | Deterministic DDIM, 100 respaced steps |
| Training subset | First 1,024 CIFAR-10 training images |
| Training budget | 1,400 epochs = 11,200 optimizer updates |

This is a target-model change. Carlini's generate-and-filter implementation and thresholds are unchanged.
[Carlini et al., Section 5.1](https://www.usenix.org/system/files/usenixsecurity23-carlini.pdf) cite
OpenAI's Improved Diffusion implementation, but train on random halves of CIFAR-10. Our prefix subset,
training budget, and attack generation budget differ. This example does not reproduce their extraction
rates or the [Improved DDPM paper's](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)
reported FID. Its released CIFAR-10 checkpoint was trained for 500,000 updates.

The `smoke` profile uses a smaller official U-Net and one epoch. It checks execution only.
The demonstration uses a shorter budget and EMA decay than the published training run. On the remote
RTX 2080 Ti, two measured full CIFAR-10 epochs with the actual training helper took 8.64 and 8.12 seconds
at microbatch 32. This projects to about 3.2-3.4 hours for 1,400 epochs. It is an estimate, not a hard
four-hour limit or evidence of convergence. The released-checkpoint download and both audits add time.
At 11,200 updates, EMA decay 0.999 retains about 0.0014% weight from initialization; decay 0.9999
would retain about 32.6%. Inspect unguided samples before interpreting extraction results.
A low normalized pixel distance can select a blurred image with similar colors.

## Sampling checks

Changing `sampling_steps` reuses the trained weights. Compare 100 and 1,000 steps with the same seed
using the notebook's unguided preview. The noise schedule remains 4,000 training steps in both cases.
The sampler uses OpenAI's respaced DDIM equations, including epsilon recomputation after clipping.
SIDE receives the original integer diffusion timestep for its classifier and forward-noising process.
After SIDE guidance, this example clips the clean prediction again to `[-1, 1]` and updates
epsilon consistently. This is a deliberate stabilization change from OpenAI's score guidance,
not a paper-defined step. The additive update avoids cancellation at near-zero alpha and
preserves unconditional sampling exactly when guidance is zero.

## Released checkpoint and training state

The first stage uses `ensure_released_checkpoint(path)` and `load_pretrained_target(profile, path, device)`.
The official [CIFAR-10 hybrid-loss checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt)
is downloaded if needed and verified with SHA-256
`351ef8b22e54e0eb2f8ef70389e101ef67e313f8fda41b46139575b3ce042cec`. A corrupted cache is rejected.
The `released` section of `train_config.yaml` sets its cache path and sampling steps.
`LEAKPRO_CIFAR_RELEASED_CHECKPOINT` can point to an existing copy, which is also hash-verified.
Its architecture is always the published 128-channel, three-block, 4,000-step configuration,
even when the scratch profile is `smoke`.

That checkpoint was trained on all 50,000 CIFAR-10 training images. The notebook audits only the
selected reference prefix and records this partial coverage. The fresh target trains only on the
selected prefix and never initializes from the released weights. Reusing an existing compatible
checkpoint in the `from_scratch` directory is supported.

The notebook records checkpoint, model source, notebook source, sampling steps, reference data, and
SIDE feature identities in the audit fingerprint. Old toy-model checkpoints are incompatible; keep
them in a separate target directory or explicitly request fresh training.

Training writes a separate `.resume.pt` file every 100 completed epochs and resumes from it on rerun.
The final target checkpoint is written only after the configured training budget completes. Resume
requires matching training settings and device type; it includes optimizer, online and EMA weights,
and random-generator states. `force_retrain: true` discards both final and resume checkpoints.
Keep a separate `target_dir` when you want to preserve an earlier experiment.

Run the extraction tests after installing the example requirements:

```bash
python -m pytest -q leakpro/tests/extraction_attacks
```

The CIFAR-10 helper tests skip when the optional Improved Diffusion package is absent.
