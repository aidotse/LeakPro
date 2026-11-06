# Gradient Inversion Attack (GIA) Examples

Examples of gradient inversion attacks, which reconstruct a client's private training data from the gradient it shares in federated learning. Start with [getting_started](./getting_started/README.md).

Run every command from the repository root. `--quick` is an offline CPU smoke run on synthetic data (seconds each; it checks the pipeline, not the quality). Without it the example runs the full experiment, usually with a GPU and a one-off dataset download. Results are written to `examples/gia/outputs/<script>/`.

| Section | Example | Question | Attack | Quick | Full | Needs |
|---|---|---|---|---|---|---|
| [getting_started](./getting_started/README.md) | `quickstart` | How do I run an attack? | Inverting Gradients | `python -m examples.gia.getting_started.quickstart --quick` | `python -m examples.gia.getting_started.quickstart` | CIFAR-10 (full) |
| [getting_started](./getting_started/README.md) | `compose_your_own_attack` | How do I build an attack from parts? | Cosine matching + TV + custom loss | `python -m examples.gia.getting_started.compose_your_own_attack --quick` | `python -m examples.gia.getting_started.compose_your_own_attack` | CIFAR-10 (full) |
| [getting_started](./getting_started/README.md) | `hpo_and_split_runs` | How do I tune and run stages separately? | Inverting Gradients | `python -m examples.gia.getting_started.hpo_and_split_runs --quick` | `python -m examples.gia.getting_started.hpo_and_split_runs` | `optuna`, CIFAR-10 (full) |
| [image_classification](./image_classification/README.md) | `compare_attacks` | Which classic attack is best? | DLG, iDLG, Geiping, Huang, Yin, GIA Running/Estimate | `python -m examples.gia.image_classification.compare_attacks --quick` | `python -m examples.gia.image_classification.compare_attacks` | CIFAR-10 (full) |
| [image_classification](./image_classification/README.md) | `bn_statistics` | What do BN statistics buy? | Huang, GIA Running/Estimate | `python -m examples.gia.image_classification.bn_statistics --quick` | `python -m examples.gia.image_classification.bn_statistics` | CIFAR-10 (full) |
| [image_classification](./image_classification/README.md) | `label_inference` | Which label inference works? | Oracle, iDLG, Dimitrov | `python -m examples.gia.image_classification.label_inference --quick` | `python -m examples.gia.image_classification.label_inference` | CIFAR-10 (full) |
| [image_classification](./image_classification/README.md) | `multi_seed` | Does multi-seed help? | See-Through-Gradients | `python -m examples.gia.image_classification.multi_seed --quick` | `python -m examples.gia.image_classification.multi_seed` | CIFAR-10 (full) |
| [image_classification](./image_classification/README.md) | `fedavg` | Effect of local epochs and batch size? | Dimitrov | `python -m examples.gia.image_classification.fedavg --quick` | `python -m examples.gia.image_classification.fedavg` | CIFAR-10 (full) |
| [image_classification](./image_classification/README.md) | `scaling` | Quality versus batch size and epochs? | Inverting Gradients (batch), Dimitrov (epochs) | `python -m examples.gia.image_classification.scaling --quick` | `python -m examples.gia.image_classification.scaling` | CIFAR-10 (full) |
| [image_classification](./image_classification/README.md) | `optimization_choices` | Which optimisation choices matter? | Inverting Gradients | `python -m examples.gia.image_classification.optimization_choices --quick` | `python -m examples.gia.image_classification.optimization_choices` | CIFAR-10 (full) |
| [generative_priors](./generative_priors/README.md) | `gan_priors` | Does a GAN prior help? | GIAS, GIFD, GGL | `python -m examples.gia.generative_priors.gan_priors --quick` | `python -m examples.gia.generative_priors.gan_priors` | CIFAR-10 DCGAN (full); `cma` for GGL |
| [generative_priors](./generative_priors/README.md) | `diffusion_priors` | Does a diffusion prior help? | GGDM, GradInvDiff | `python -m examples.gia.generative_priors.diffusion_priors --quick` | `python -m examples.gia.generative_priors.diffusion_priors` | `diffusers`; DDPM (full) |
| [object_detection](./object_detection/README.md) | `detection_attack` | Can images and boxes be recovered from a detector? | Detection GIA (DETR, RetinaNet) | `python -m examples.gia.object_detection.detection_attack --detector detr --quick` | `python -m examples.gia.object_detection.detection_attack --detector detr` | `transformers` (full) |
| [tabular](./tabular/README.md) | `tabular_attack` | Can tabular rows be recovered? | Softmax relaxation, raw one-hot attack | `python -m examples.gia.tabular.tabular_attack --quick` | `python -m examples.gia.tabular.tabular_attack` | - |
| [tabular](./tabular/README.md) | `california_housing` | What leaks about housing blocks? | Softmax relaxation | `python -m examples.gia.tabular.california_housing --quick` | `python -m examples.gia.tabular.california_housing` | California Housing (full) |
| [tabular](./tabular/README.md) | `leakage_factors.architecture` | Which architecture factors drive leakage? | Raw one-hot attack | `python -m examples.gia.tabular.leakage_factors.architecture --quick` | `python -m examples.gia.tabular.leakage_factors.architecture` | Adult, California Housing (full) |
| [tabular](./tabular/README.md) | `leakage_factors.attack_iterations` | Does a larger budget help? | Raw one-hot attack | `python -m examples.gia.tabular.leakage_factors.attack_iterations --quick` | `python -m examples.gia.tabular.leakage_factors.attack_iterations` | Adult, California Housing (full) |
| [tabular](./tabular/README.md) | `leakage_factors.batch_sizes` | Leakage versus batch size? | Raw one-hot attack | `python -m examples.gia.tabular.leakage_factors.batch_sizes --quick` | `python -m examples.gia.tabular.leakage_factors.batch_sizes` | Adult, California Housing (full) |
| [tabular](./tabular/README.md) | `leakage_factors.fedavg` | Local epochs versus batch size? | Raw one-hot attack | `python -m examples.gia.tabular.leakage_factors.fedavg --quick` | `python -m examples.gia.tabular.leakage_factors.fedavg` | Adult, California Housing (full) |
| [tabular](./tabular/README.md) | `leakage_factors.label_knowledge` | Unknown labels? | Raw one-hot attack | `python -m examples.gia.tabular.leakage_factors.label_knowledge --quick` | `python -m examples.gia.tabular.leakage_factors.label_knowledge` | Adult, California Housing (full) |
| [tabular](./tabular/README.md) | `leakage_factors.training_exposure` | Leakage during training? | Raw one-hot attack | `python -m examples.gia.tabular.leakage_factors.training_exposure --quick` | `python -m examples.gia.tabular.leakage_factors.training_exposure` | Adult, California Housing (full) |
| [coco_yolo_1_image](./coco_yolo_1_image/README.md) | `yolo_attack` | Can a YOLO-style detector be inverted? | Detection GIA (YOLO) | `python -m examples.gia.coco_yolo_1_image.yolo_attack --quick` | `python -m examples.gia.coco_yolo_1_image.yolo_attack --device cuda` | none (synthetic) |

The other folders here (`inverting_cifar10_1_image`, `inverting_cifar100_16_images`, `inverting_celebA_1_image`, `huang_cifar10_16_images`, `huang_optuna`, `GIA_base`, `GIA_base_running`, `GIA_base_convnext_imagenet`, `coco_yolo_1_image` (its `main.py`), `pii_inverting_masked_text` and `evaluating_gia_train`) predate the modular framework: they use the legacy attack classes in `leakpro/attacks/gia_attacks/*.py`, and each has its own README.

The `common/` folder holds the shared command-line, data, model, runner and plotting helpers used by the examples.

## Reproducing the results

Each section README has a results block per comparative script, generated from the saved `results.json` files. To rerun every example in the quick mode as the test suite does:

```
pytest leakpro/tests/gia/integration/test_examples_quick.py --run-slow
```

To rerun every example at full size, with the settings the READMEs report:

```
python -m examples.gia.common.run_all                 # everything not yet done
python -m examples.gia.common.run_all --only fedavg   # selected examples
python -m examples.gia.common.run_all --dry-run       # print the commands only
```

`run_all` writes one directory per example under `examples/gia/outputs/`, logs each run to its `run.log`, waits for a free GPU, continues after a failed example, and skips examples whose `results.json` was written at the current commit by a run that succeeded (`--any-commit` accepts results from any commit). Then render the result blocks into a README with `python -m examples.gia.common.report <README.md>`.

Every example accepts two flags that matter for full runs:

- `--iter-scale F` multiplies every optimisation budget by `F` (for example `0.25` for a shorter run on a slow GPU). `--quick` ignores it.
- `--deterministic` makes the run bit-reproducible on one machine: deterministic CUDA kernels and cuDNN without autotuning. It is slower, and `run_all` always sets it. Without it, GPU runs of the same command differ by up to a couple of dB in PSNR.
