# DETR gradient inversion — session handoff

Status as of 2026-08-13. Goal: reproduce the `coco_yolo_1_image` experiment for DETR. Step 1 (get the
GIA attack running against DETR) is **done**. The attack executes end to end and is correct, but on
stock DETR **it recovers nothing** — every configuration tried sits at SSIM ~0.002, i.e. noise.

Nothing here is committed. `git status` shows this directory as untracked plus two modified core files.

## Two corrections to the earlier framing

**The YOLO comparison was not apples to apples.** `coco_yolo_1_image/main.py` is *randomly
initialized* (`yolo_v8_n_basicblock()` takes no weights) and reaches its result through
`run_with_optuna` with **100 trials** searching `total_variation`, `attack_lr`, `median_pooling`,
`top10norms` *and* which of 5 candidate images to attack. Its best value was **SSIM 0.355**, while
individual trials in the tail sit at 0.013–0.018. Every DETR number in the table below comes from a
single hand-picked setting with `attack_lr` fixed at 0.1, and optuna searches that parameter over
1e-4 to 100 (log). So "DETR does not leak" was, until now, undertested rather than established.
`run_optuna.py` applies the YOLO protocol to DETR.

**Random init is not inherently hopeless** — it is hopeless *for stock-depth DETR*. The YOLO example
proves a randomly initialized shallow detector can leak. What kills stock DETR is where the update
norm lands, not the absence of pretrained weights.

## Run it

**Use the system `python3` (3.10), NOT the project `.venv`.** The venv's `pytorch-ignite` segfaults on
import under torch 2.10, and `leakpro/fl_utils/similarity_measurements.py` imports ignite at module
level, so every GIA attack dies with a bare core dump and no traceback. This is not specific to DETR.
The box also has an unrelated NVML driver/library mismatch that makes `nvidia-smi` fail while CUDA
itself works — don't be misled by it.

```bash
cd examples/gia/coco_detr_1_image   # relative COCO/ paths require this cwd
python3 main.py                      # full 8000 iteration attack, ~45 min

# quick probe; --model small --variant X selects the configurable DETR instead of the stock one
python3 run_experiments.py --name probe --iters 750 --unfreeze-bn 1 --attack base
python3 run_experiments.py --name probe --model small --variant r18-notrans --iters 750

# diagnostics
python3 layer_analysis.py --pretrained 1 --unfreeze-bn 1        # where the update norm and signal are
python3 matching_stability.py --pretrained 1 --unfreeze-bn 1    # does Hungarian matching flip?

# the fair comparison: same protocol the YOLO example used to get its result
python3 run_optuna.py --name opt --model small --variant r18-notrans --attack base \
    --trials 30 --iters 2000 --check-interval 250 --min-objects 10 --max-objects 100
```

All scripts share `--model/--variant/--pretrained/--unfreeze-bn` (see `build.py:add_model_args`) and
`--img-size/--min-objects/--max-objects/--start-idx/--split/--image-id/--letterbox` (`add_data_args`).

To attack the exact image the YOLO paper figure uses, add `--split train2017 --image-id 531715`.

COCO val2017 is already downloaded to `COCO/` (gitignored). Extra deps beyond the `federated` extra:
`transformers`, `timm`, `pycocotools` — installed in both envs already.

Runtime: stock DETR 0.34 s/iter at 256x256, 3 GB peak on the Quadro RTX 5000; the small variants
around 0.19 s/iter.

Ignore two loud but harmless warnings when loading the stock checkpoint: `num_batches_tracked` keys
"not used", and a wall of "copying from a non-meta parameter ... is a no-op" from timm's backbone
construction path. The backbone weights **are** loaded — verified byte-identical against
`model.safetensors`. Don't spend time on it again.

## What was built

- `coco.py` — COCO to DETR targets. Normalized `cxcywh` boxes, **raw COCO `category_id`** as class label
  (that is the pretrained checkpoint's 91 class label space), ImageNet normalization so the attack's
  pixel clamp maps exactly onto the valid image range, and YOLO-identical letterboxing (see below).
- `model.py` — `detr_resnet50()`, `ComputeLoss` (wraps the transformers Hungarian set prediction loss),
  and `unfreeze_batch_norm()` (see below).
- `main.py`, `download_coco.sh`, `README.md`.
- `run_experiments.py`, `layer_analysis.py` — the diagnostics used for the results below.
- `detr_small.py` — **a compact, hackable DETR** (see below), the vehicle for architecture surgery.
- `build.py` — shared `--model`/`--variant`/`--img-size`/`--min-objects` flags and builders, so the
  runner, the optuna search and every diagnostic construct identical models and loaders.
- `run_optuna.py` — the YOLO example's hyperparameter search protocol, applied to DETR.
- `matching_stability.py` — tracks whether the Hungarian assignment flips during the attack.
- `logs/run_*.sh` — the exact sweeps that produced the tables below, plus their logs.

### Letterboxing, so the two examples attack the same pixels (`--letterbox`, default on)

Every result table below predates this and was produced with the old behaviour, a plain
`transforms.Resize((256, 256))` that **stretches** a non-square COCO image to the square. The YOLO
example instead **letterboxes**: it scales the longest side to 256 preserving the aspect ratio and
centres the result on a black canvas, so its figures have black bars. A paper putting the two
reconstructions side by side needs them preprocessed identically, so `coco.py` now letterboxes too.

`letterbox_geometry`/`letterbox_image`/`letterbox_boxes` replicate `coco_yolo_1_image/coco.py`'s
arithmetic exactly, down to the truncation in `load_image`, the `round(pad ± 0.1)` split of an odd
number of padding rows, and `xy2wh`'s clip of boxes to the padded canvas. Two things that are easy to
get wrong and are worth not rediscovering:

- **Pad in raw pixel space, before normalization.** The bars must be 0 in `[0,1]`, i.e. `-mean/std`
  after normalization, because `img_save` writes `clamp(x*std + mean, 0, 1)`. Padding the normalized
  tensor with zeros instead gives grey bars. `-mean/std` is also exactly the attack's lower pixel clamp.
- **Resample with `cv2.INTER_LINEAR`, not a torchvision resize.** PIL's antialiased bilinear differs
  from cv2's on this 2.5x downscale by up to 110/255 (mean 3.3), which would leave the two examples'
  "ground truth" visibly different. JPEG decoding is identical between the two libraries (verified
  max diff 0), so `Image.open` is still fine for reading.
- **Boxes have to move.** Under a stretch, normalized coordinates are invariant, which is why the old
  code could normalize by the original width and height and stop there. Letterboxing shifts and shrinks
  the content, so the boxes must be mapped onto the canvas.

Verified on the paper's skateboard image (train2017 id 531715, 640x427): both pipelines produce 43
black rows at the top, 170 rows of content, 43 at the bottom, and
`max|detr_raw - yolo| = 6e-08` over the whole tensor with all 5 boxes agreeing to 0.0. The
reconstruction is *not* masked to the content region in either example — the attack has to recover the
bars, same as YOLO.

`--letterbox 0` restores the stretch, which is what to use when reproducing a number from the tables
below.

### `detr_small.py`

A single-file reimplementation of the original DETR (Carion et al.), faithful in structure and forward
pass but with every architectural choice a constructor argument, so it can be cut down the way the YOLO
example cut down YOLO. Two deliberate departures: BatchNorm is ordinary trainable `nn.BatchNorm2d`
rather than frozen buffers, and the encoder/decoder stacks may have **zero** layers, in which case the
heads read backbone feature-map cells directly and DETR degenerates into a dense detector. Parameter
names mirror `transformers.DetrForObjectDetection`, so `layer_analysis.py` groups both models with the
same code. Named points on the ladder live in `VARIANTS`; verified to build, produce a differentiable
set-prediction loss, and pass second-order gradients through `MetaModule` to the pixels.

Why write one rather than vendor `facebookresearch/detr`: that repo spreads the model over six files
plus distributed-training utilities that need stubbing, and its `SetCriterion` would then have to be
reconciled with the transformers loss the stock path already uses. This keeps one loss for both models.

### `FixedMatchComputeLoss` (in `model.py`, flag `--fixed-match 1`)

DETR's loss re-solves Hungarian matching on every forward pass, so as the attack moves the pixels the
query-to-object assignment changes, and each assignment is a *different loss* — a different gradient for
the attack to chase. This loss freezes the assignment to the one the true client image induces.

Why that is legitimate rather than cheating the metric: at the true image the frozen assignment **is**
what Hungarian matching returns, so an honest client's shared update is unchanged. Verified numerically
— `max|grad_hungarian - grad_fixed| = 0.000e+00` exactly, on a 20 object image — while the two
objectives genuinely differ away from the true image (14.22 vs 16.30 at a noise image). The reconstruction
target and the global optimum are untouched; only the attacker's path through pixel space is smoothed.
Obtaining that assignment is oracle knowledge, so this measures *whether matching instability is the
blocker*. `--refresh-every N` is the deployable version: re-solve every N calls instead of never.

Core additions (DETR targets are per image dicts of varying length, so neither the tensor nor the YOLO
path fits):
- `leakpro/fl_utils/data_utils.py` — `detr_collate_fn`, `CustomDetrTensorDataset`, `GiaImageDetrExtension`
- `leakpro/fl_utils/gia_train.py` — `train_detr`

Also fixed: `data_utils.py:8` imported `Self` from `typing`, which is 3.11+ only while `pyproject.toml`
declares `>=3.8`. It made the module unimportable on python3.10. `import_helper` already provided a
version compatible `Self` two lines below.

## A confound in every result below: all of them attacked a one-object image

`run_experiments.py` defaulted to `max_objects=3` and the loader's `min_objects` was 1, so it picked the
first val2017 image in that range — **image 285, which has exactly one annotated object** (class 23).
Every number in the table below therefore comes from an image where 1 of 100 queries received positive
box and class supervision and the other 99 contributed only "no object" classification loss, against
196k free pixels. Object count is now a first class flag (`--min-objects/--max-objects`); the first
val2017 image has 20 objects.

## Results — stock DETR, one-object image, all SSIM ~0.002

| model | attack / objective | best loss | SSIM |
|---|---|---|---|
| pretrained | inverting, global cosine, tv=1e-2 | 0.406 | 0.0020 |
| pretrained | inverting, global cosine, tv=1e-4 | 0.477 | 0.0028 |
| random init | inverting, tv=1e-2 | 0.0004 | 0.0088 |
| random init | inverting, tv=1e-4 | 0.0006 | 0.0081 |
| pretrained | per-layer normalized cosine | 0.894 | 0.0019 |
| pretrained | backbone parameters only | 0.957 | 0.0021 |
| pretrained | per-layer + signed gradients | 0.850 | 0.0018 |
| pretrained + BN unfrozen | inverting, global cosine | 0.488 | 0.0020 |
| pretrained + BN unfrozen | GIABaseRunning (signed + BN reg) | **0.273** | 0.0023 |

Every run plateaus permanently (`best_loss` stops improving) by iteration 250–750 and never recovers.

## Results — 2026-08-13 session

### Step 2: lower resolution does not help, it hurts

Stock DETR, pretrained, BN unfrozen, 1500 iterations, one-object image.

| resolution | best loss | SSIM |
|---|---|---|
| 256 px | 0.488 | 0.0020 |
| 128 px | 0.918 | 0.0036 |
| 64 px | 0.942 | 0.0055 |

At 64 px the loss freezes at **iteration 22** and never moves again; at 128 px almost as early. Fewer
free variables did not make the objective easier — it made it undescendable. Resolution was not the
constraint. Don't revisit.

### Step 3: the architecture ladder does work

`detr_small.py` variants, ImageNet backbone, one-object image, 1500 iterations, tv=1e-2, lr=0.1.
Stock DETR's best-ever cell for comparison: loss 0.273, SSIM 0.0023.

| variant | what changed | best loss | SSIM |
|---|---|---|---|
| `r18-notrans` | no transformer at all, dense heads on stride-16 backbone | 0.105 | **0.0241** |
| `r18-s3` | stock 6+6 transformer, stride-16 ResNet-18 backbone | 0.015 | 0.0164 |
| `r18-shallow-s3` | 1+1 transformer, d=128, stride-16 backbone | 0.015 | 0.0123 |
| `r18-nodec` | 1 encoder, no decoder | 0.118 | 0.0065 |
| `r18-shallow` | 1+1 transformer, d=128, stride-32 backbone | 0.205 | 0.0036 |
| `r18` | stock depth, ResNet-18 backbone, trainable BN | 0.106 | 0.0027 |

**Roughly 10x the stock model, and the ordering points at the transformer.** The best cell has no
transformer at all; the worst is stock-depth. Two secondary reads: a stride-16 backbone beats stride-32
at equal transformer depth (`r18-s3` 0.0164 vs `r18` 0.0027, `r18-shallow-s3` 0.0123 vs `r18-shallow`
0.0036), and low loss does not imply high SSIM — `r18-s3` and `r18-shallow-s3` both reach loss 0.015
with quite different SSIM, so the objective is still only loosely tied to pixel fidelity.

For scale: the YOLO example's *untuned* trials sit at SSIM 0.013–0.018 and it reached 0.355 only after
100 optuna trials. The ladder's top cells are at untuned-YOLO parity, so a search is the obvious next
lever, not more surgery.

### More objects makes stock DETR strictly worse

Stock DETR, pretrained, BN unfrozen, 1500 iterations. This tests the natural reading of the confound
above — more annotated objects means more queries carrying positive supervision.

| objects in image | best loss | SSIM |
|---|---|---|
| 1 | 0.488 | 0.0020 |
| 7 | 0.947 | 0.0006 |
| 20 | 0.924 | 0.0028 |

The extra supervision does not arrive as extra leakage: the loss goes from descending (0.488) to nearly
frozen (0.92–0.95). Whatever the added objects contribute in signal, they cost more in optimizability.
The next section says why.

### Hungarian matching is unstable, and that is DETR specific

`matching_stability.py` walks a straight line in pixel space from noise to the true image and reports
how the query-to-object assignment moves. Stock DETR, pretrained, BN unfrozen, 20 steps:

| image | reassignments | rate | still mismatched at 95% of the way |
|---|---|---|---|
| 1 object | 2 / 20 | 10% | 0 / 1 |
| 20 objects | 238 / 400 | **59.5%** | **9 / 20** |

On a crowded image the assignment is in constant churn, and it has *not* converged even immediately
next to the true image — the final 0.95 to 1.00 step alone flips 9 of 20 objects. Each assignment
defines a different loss, so the attack is matching a gradient whose definition keeps changing under
it. This is a mechanism no CNN detector has, and it explains both the permanent plateau and why more
objects hurt.

Caution for whoever reads this next: the one-object image says "matching is stable" (10%, converges by
alpha=0.20). That conclusion is an artifact of having a single object. Always check this on a crowded
image.

`FixedMatchComputeLoss` / `--fixed-match 1` is the test of whether this is *the* blocker; those runs
were still in flight when this was written. See the top of the file for why freezing the assignment
leaves the honest client's update bit-identical.

## Diagnosis — two different failure modes

From `layer_analysis.py`, which reports per parameter group: share of the update norm, cosine of the
client gradient against the gradient from a **noise** image, and against a **different real** image.

**Random init is hopeless.** 99.97% of the update norm is in the decoder and *every* group has
cos(noise) ~0.998. The update carries essentially no image information, so the attack drives the
objective to 0.0004 while learning nothing. Round-0 FL is a dead end for DETR; do not spend more time
on random init.

**Pretrained has signal but won't optimize.** The backbone gradient has cos(other real image) = 0.032,
so it is strongly image specific. But 97.7% of the update norm is in the box head.

```
group         #par  norm share  cos(noise)  cos(other)     (pretrained, BN unfrozen)
backbone        53      0.1837      0.0203      0.0318
backbone_bn    106      0.0501      0.0607     -0.0582
encoder         96      0.0873      0.0232      0.1950
decoder        158      0.0646      0.0081      0.2430
cls_head         2      0.0031     -0.8003     -0.1564
box_head         6      0.9755     -0.0293      0.8008
ALL            424      1.0000      0.0024      0.4828
```

## Ruled out — do not redo these

- **The weighting hypothesis is disproven.** The obvious read of the table above is that the box head's
  97.7% norm share drowns out the pixel bearing backbone. It does not explain the failure: restricting
  the objective *exclusively* to backbone parameters still cannot descend (1.002 -> 0.957 in 500 iters).
- Per-layer normalized cosine, signed gradients, tv_reg 1e-2 vs 1e-4, median pooling on/off.

## The frozen BatchNorm question (investigated, partially resolved)

`DetrFrozenBatchNorm2d` computes `x*w/sqrt(var+eps) + (b - mean*w/sqrt(var+eps))`, which is
**algebraically identical** to `nn.BatchNorm2d` in eval mode, same eps=1e-5. Verified: converting all 53
layers changes logits by 2e-5 and boxes by 3e-6 (float32 noise). The detector is unchanged.

But it stores `weight`/`bias` as **buffers, not Parameters**, which matters a lot for GIA:
1. They contribute zero entries to the shared update. Conversion takes it from 318 to 424 tensors
   (+53x2) carrying 5.0% of the update norm — signal stock DETR never shares.
2. Those gradients are strongly image specific (cos = -0.058 against a different real image).
3. LeakPro's `isinstance(module, nn.BatchNorm2d)` checks now match, so the BN machinery activates
   instead of silently doing nothing.

`unfreeze_batch_norm()` in `model.py` does this conversion.

**Result: it was not enough.** With BN unfrozen the plateau moves later (0.653 at 250, 0.488 at 500) but
still lands at SSIM 0.0020, identical to frozen. `GIABaseRunning` on the unfrozen model reaches
**loss 0.273 — the lowest of any pretrained run, roughly half the frozen BN best** — confirming the BN
gradients really are extra usable signal. It still plateaus by iteration 250 at SSIM 0.0023. So the added
signal is real but the optimizer cannot convert it into pixels.

Caveat on the BN regularizer: in **eval** mode running stats don't move, so `GIABaseRunning`'s
`used_mean = 10*rm_post - 9*rm_pre` collapses to the model's stored global statistics — a
DeepInversion style realism prior, not client specific leakage. In **train** mode the stats do move and
that formula recovers the client's real per channel batch statistics, which at batch size 1 *are* the
image's statistics. That is a strictly stronger threat model and the regime the attack was built for.
**Untested — probably the highest value next experiment.**

## Suggested next steps, in order

1. **Optuna on the best ladder variants.** This is the biggest untested lever and the only way the
   comparison to YOLO is honest — that example's result came from 100 trials, and every DETR number to
   date used `attack_lr = 0.1` while the search range is 1e-4 to 100. Start with `r18-notrans` and
   `r18-s3`, `--attack base`, 25-30 trials, `--iters 3000 --check-interval 250`. Budget ~2-3 h per
   variant. `logs/run_optuna_sweep.sh` has a ready sweep (written before the ladder finished, so it
   targets `r18-notrans` and `r18` — add `r18-s3`).
2. **Finish reading the `--fixed-match` results** (in flight as of this writing; `logs/fm_*.log`). If
   freezing the assignment unblocks the plateau, the practical follow-up is `--refresh-every N` tuning,
   and the finding is "DETR's set-prediction loss, not its depth, is what resists inversion". If it does
   not, matching instability is a real property but not the binding constraint, and the story reverts to
   architecture depth.
3. **Train the shallow variants briefly before attacking.** The ladder's variants have an ImageNet
   backbone but a randomly initialized transformer and heads, and that shows: `r18` drives the objective
   to 0.106 while recovering nothing, i.e. a wrong image satisfies the gradient match. A few thousand
   COCO steps would put them in the regime where the objective's minimum actually sits at the true
   image. This is the most likely route to a YOLO-comparable number and nothing here has tested it.
4. Client in **train** mode with BN unfrozen, so `GIABaseRunning` can infer real batch statistics.
   Check `MetaModule`'s buffer handling propagates the in place running stat updates. Note this matters
   much less for the `detr_small` variants, whose BatchNorm is trainable by construction. Skipped this
   session by choice, not because it was answered.
5. A negative result is publishable, and it is now a much better one than "DETR resists inversion".
   The measured claims are: the update norm concentrates 95% in the box head; the shallow ladder buys
   ~10x and points at the transformer; resolution reduction actively hurts; more objects hurt; and the
   bipartite matching reassigns 60% of objects per step on a crowded image, which is a failure mode
   unique to set prediction detectors. Each is backed by a script in this directory.

## Things that look like bugs but are not

- The stock checkpoint load prints `num_batches_tracked` "not used" plus a wall of "copying from a
  non-meta parameter ... is a no-op". Backbone weights **are** loaded, verified byte-identical against
  `model.safetensors`.
- `nvidia-smi` fails with an NVML mismatch while CUDA works fine.
- `matching_stability.py` reporting a stable assignment — check the object count before believing it.
