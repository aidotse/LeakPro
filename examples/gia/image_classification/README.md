# Image classification

Comparative experiments on image classifiers. Run from the repository root; `--quick` is an offline CPU smoke run on synthetic data (a few seconds to about half a minute), the default is the full experiment (CIFAR-10 is downloaded once to `data/`; a GPU is recommended). Results go to `examples/gia/outputs/<script>/`.

Each script has the form `python -m examples.gia.image_classification.<script> [--quick]`.

## compare_attacks
Which classic attack recovers images best from one gradient? DLG, iDLG, Inverting Gradients, Huang, Yin, GIA Running and GIA Estimate.

<!-- results:compare_attacks -->
| Attack | psnr | ssim |
|---|---|---|
| DLG | 8.02 ± 0.95 (n=4) | 0.01 ± 0.01 (n=4) |
| iDLG | 14.00 ± 4.18 (n=4) | 0.26 ± 0.18 (n=4) |
| Inverting Gradients | 19.70 ± 8.41 (n=4) | 0.60 ± 0.28 (n=4) |
| Huang | 13.06 ± 2.09 (n=4) | 0.19 ± 0.07 (n=4) |
| Yin | 11.81 ± 1.47 (n=4) | 0.07 ± 0.02 (n=4) |
| GIA Running | 13.60 ± 2.57 (n=4) | 0.20 ± 0.10 (n=4) |
| GIA Estimate | 13.12 ± 2.10 (n=4) | 0.17 ± 0.08 (n=4) |

Command: `python -m examples.gia.image_classification.compare_attacks --out examples/gia/outputs/compare_attacks --num-images 4 --seeds 1 --iter-scale 0.2`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest PSNR: Inverting Gradients (19.70 dB); lowest: DLG (8.02 dB).

## bn_statistics
What does knowing the BatchNorm statistics buy? Running versus proxy-estimated statistics, with the client in train or eval mode.

<!-- results:bn_statistics -->
| Attack | psnr | ssim |
|---|---|---|
| No BN knowledge (Inverting Gradients) | 10.41 ± 1.62 (n=12) | 0.04 ± 0.02 (n=12) |
| Huang (client running statistics) | 12.95 ± 1.97 (n=12) | 0.17 ± 0.06 (n=12) |
| GIA Running | 13.36 ± 2.28 (n=12) | 0.18 ± 0.08 (n=12) |
| GIA Estimate (proxy data) | 13.25 ± 2.12 (n=12) | 0.19 ± 0.07 (n=12) |

Command: `python -m examples.gia.image_classification.bn_statistics --out examples/gia/outputs/bn_statistics --num-images 4 --seeds 3 --iter-scale 0.2`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest PSNR: GIA Running (13.36 dB); lowest: No BN knowledge (Inverting Gradients) (10.41 dB).

## label_inference
How many labels does each label-inference method recover as the client's batch grows?

`multiset_accuracy` is the fraction of the true label multiset that the method recovers, which is the meaningful score. `accuracy` compares labels image by image after matching the reconstructed images to the originals; this script runs the attack for a single step, so that matching is arbitrary and `accuracy` only reflects it (it is low even for the oracle labels). Read `multiset_accuracy`.

<!-- results:label_inference -->
| Attack | accuracy | multiset_accuracy |
|---|---|---|
| bs=1 / Oracle | 1.00 ± 0.00 (n=20) | 1.00 ± 0.00 (n=20) |
| bs=1 / iDLG | 1.00 ± 0.00 (n=20) | 1.00 ± 0.00 (n=20) |
| bs=1 / Geng | 1.00 ± 0.00 (n=20) | 1.00 ± 0.00 (n=20) |
| bs=1 / Dimitrov (start) | 1.00 ± 0.00 (n=20) | 1.00 ± 0.00 (n=20) |
| bs=1 / Dimitrov (end) | 1.00 ± 0.00 (n=20) | 1.00 ± 0.00 (n=20) |
| bs=1 / Dimitrov (avg) | 1.00 ± 0.00 (n=20) | 1.00 ± 0.00 (n=20) |
| bs=2 / Oracle | 0.60 ± 0.49 (n=20) | 1.00 ± 0.00 (n=20) |
| bs=2 / iDLG | 0.45 ± 0.44 (n=20) | 0.90 ± 0.20 (n=20) |
| bs=2 / Geng | 0.60 ± 0.20 (n=20) | 0.60 ± 0.20 (n=20) |
| bs=2 / Dimitrov (start) | 0.60 ± 0.20 (n=20) | 0.60 ± 0.20 (n=20) |
| bs=2 / Dimitrov (end) | 0.60 ± 0.20 (n=20) | 0.60 ± 0.20 (n=20) |
| bs=2 / Dimitrov (avg) | 0.60 ± 0.20 (n=20) | 0.60 ± 0.20 (n=20) |
| bs=4 / Oracle | 0.28 ± 0.25 (n=20) | 1.00 ± 0.00 (n=20) |
| bs=4 / iDLG | 0.29 ± 0.23 (n=20) | 0.84 ± 0.16 (n=20) |
| bs=4 / Geng | 0.38 ± 0.17 (n=20) | 0.38 ± 0.17 (n=20) |
| bs=4 / Dimitrov (start) | 0.38 ± 0.17 (n=20) | 0.38 ± 0.17 (n=20) |
| bs=4 / Dimitrov (end) | 0.38 ± 0.17 (n=20) | 0.38 ± 0.17 (n=20) |
| bs=4 / Dimitrov (avg) | 0.38 ± 0.17 (n=20) | 0.38 ± 0.17 (n=20) |
| bs=8 / Oracle | 0.20 ± 0.13 (n=20) | 1.00 ± 0.00 (n=20) |
| bs=8 / iDLG | 0.16 ± 0.12 (n=20) | 0.71 ± 0.13 (n=20) |
| bs=8 / Geng | 0.20 ± 0.11 (n=20) | 0.24 ± 0.10 (n=20) |
| bs=8 / Dimitrov (start) | 0.21 ± 0.09 (n=20) | 0.22 ± 0.10 (n=20) |
| bs=8 / Dimitrov (end) | 0.21 ± 0.09 (n=20) | 0.23 ± 0.10 (n=20) |
| bs=8 / Dimitrov (avg) | 0.21 ± 0.09 (n=20) | 0.23 ± 0.10 (n=20) |
| bs=16 / Oracle | 0.18 ± 0.09 (n=20) | 1.00 ± 0.00 (n=20) |
| bs=16 / Geng | 0.17 ± 0.05 (n=20) | 0.21 ± 0.08 (n=20) |
| bs=16 / Dimitrov (start) | 0.17 ± 0.05 (n=20) | 0.17 ± 0.05 (n=20) |
| bs=16 / Dimitrov (end) | 0.17 ± 0.05 (n=20) | 0.17 ± 0.05 (n=20) |
| bs=16 / Dimitrov (avg) | 0.17 ± 0.05 (n=20) | 0.17 ± 0.05 (n=20) |

Command: `python -m examples.gia.image_classification.label_inference --out examples/gia/outputs/label_inference --device cuda --trials 20`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Multiset accuracy is 1.00 for every method at batch size 1 and for the oracle at every batch size. Above batch size 1, iDLG recovers the most labels while it applies (0.90, 0.84 and 0.71 at batch sizes 2, 4 and 8); at batch size 16 it cannot label a batch larger than the number of classes, and the lowest score is Dimitrov (0.17) against Geng (0.21).

## multi_seed
Does See-Through-Gradients' multi-seed search with group consistency beat a single seed at the same budget?

<!-- results:multi_seed -->
| Attack | psnr | ssim |
|---|---|---|
| 1 seed | 11.26 ± 2.12 (n=4) | 0.07 ± 0.05 (n=4) |
| 4 seeds, averaged | 12.13 ± 1.66 (n=4) | 0.09 ± 0.04 (n=4) |
| 4 seeds, best loss | 10.48 ± 1.68 (n=4) | 0.05 ± 0.02 (n=4) |

Command: `python -m examples.gia.image_classification.multi_seed --out examples/gia/outputs/multi_seed --num-images 4 --iter-scale 0.4`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest PSNR: 4 seeds, averaged (12.13 dB); lowest: 4 seeds, best loss (10.48 dB).

## fedavg
How do the number of local epochs and the client's batch size change what Dimitrov's FedAvg attack recovers?

<!-- results:fedavg -->
| Attack | psnr | ssim |
|---|---|---|
| epochs=1, client batch=1 | 45.28 ± 0.81 (n=2) | 1.00 ± 0.00 (n=2) |
| epochs=1, client batch=2 | 45.13 ± 0.21 (n=2) | 1.00 ± 0.00 (n=2) |
| epochs=2, client batch=1 | 39.65 ± 0.30 (n=2) | 0.99 ± 0.00 (n=2) |
| epochs=2, client batch=2 | 38.87 ± 0.02 (n=2) | 0.99 ± 0.00 (n=2) |
| epochs=3, client batch=1 | 39.31 ± 0.03 (n=2) | 0.99 ± 0.00 (n=2) |
| epochs=3, client batch=2 | 38.88 ± 0.10 (n=2) | 0.99 ± 0.00 (n=2) |

Command: `python -m examples.gia.image_classification.fedavg --out examples/gia/outputs/fedavg --deterministic --num-images 2 --iter-scale 0.1`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest PSNR: epochs=1, client batch=1 (45.28 dB); lowest: epochs=2, client batch=2 (38.87 dB).

## scaling
How does reconstruction quality change with the batch size (FedSGD) and with the number of local epochs (FedAvg)?

<!-- results:scaling -->
| Attack | psnr | ssim |
|---|---|---|
| batch=1 | 32.99 ± 0.00 (n=1) | 0.97 ± 0.00 (n=1) |
| batch=2 | 32.56 ± 0.31 (n=2) | 0.97 ± 0.01 (n=2) |
| batch=4 | 30.75 ± 2.59 (n=4) | 0.93 ± 0.02 (n=4) |
| batch=8 | 24.25 ± 6.45 (n=8) | 0.83 ± 0.14 (n=8) |
| batch=16 | 18.74 ± 4.65 (n=16) | 0.62 ± 0.16 (n=16) |

Command: `python -m examples.gia.image_classification.scaling --out examples/gia/outputs/scaling --sweep batch`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest PSNR: batch=1 (32.99 dB); lowest: batch=16 (18.74 dB).

The epochs sweep (`--sweep epochs`, FedAvg with Dimitrov's attack on two images) is a separate run, `scaling_epochs`:

<!-- results:scaling_epochs -->
Results are filled in by `python -m examples.gia.common.report` after the next full run.
<!-- /results -->

## optimization_choices
One-factor ablation of the optimisation choices behind Inverting Gradients: pixel space, gradient sign, schedule and initial image.

<!-- results:optimization_choices -->
| Attack | psnr | ssim |
|---|---|---|
| space: optimise model input | 20.46 ± 7.59 (n=8) | 0.63 ± 0.25 (n=8) |
| space: optimise data space | 28.98 ± 7.06 (n=8) | 0.86 ± 0.21 (n=8) |
| grad_sign: signed gradient | 20.88 ± 7.48 (n=8) | 0.66 ± 0.22 (n=8) |
| grad_sign: raw gradient | 18.33 ± 6.71 (n=8) | 0.52 ± 0.26 (n=8) |
| schedule: no schedule | 24.74 ± 7.78 (n=8) | 0.77 ± 0.22 (n=8) |
| schedule: step decay | 19.66 ± 7.58 (n=8) | 0.59 ± 0.26 (n=8) |
| schedule: linear decay | 20.71 ± 6.84 (n=8) | 0.66 ± 0.25 (n=8) |
| initializer: uniform start | 28.76 ± 6.91 (n=8) | 0.85 ± 0.22 (n=8) |
| initializer: gaussian start | 37.09 ± 0.65 (n=8) | 0.98 ± 0.01 (n=8) |

Command: `python -m examples.gia.image_classification.optimization_choices --out examples/gia/outputs/optimization_choices --num-images 4 --seeds 2 --iter-scale 0.3`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

Highest PSNR: initializer: gaussian start (37.09 dB); lowest: grad_sign: raw gradient (18.33 dB).
