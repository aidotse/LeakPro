# Gradient Inversion Attack on DETR (COCO)

Gradient inversion attack, from **Geiping et al., 2020: _"Inverting Gradients - How Easy Is It to Break
Privacy in Federated Learning?"_**, applied to **DETR** (Carion et al., 2020) on COCO. It is the
transformer detector counterpart to the `coco_yolo_1_image` example: a single COCO image is handed to a
simulated client, the resulting model update is shared, and the attack tries to recover the image from
that update alone.

## What is different from the YOLO example

DETR is a set prediction model, so the client update looks different from a YOLO one in three ways that
the attack plumbing has to handle:

- **Targets are dicts, not tensors.** Each image carries `class_labels` and normalized `cxcywh` `boxes`
  of varying length, so batches keep the targets as a list of dicts (`detr_collate_fn`,
  `GiaImageDetrExtension` and `train_detr` in `leakpro/fl_utils/`).
- **The loss involves Hungarian matching.** The matching itself is not differentiable, but it is computed
  under `no_grad` and only selects which prediction is compared to which target, so the classification,
  L1 and GIoU terms remain differentiable back to the input pixels.
- **The backbone has no trainable BatchNorm.** DETR freezes its ResNet BatchNorm layers, so the BatchNorm
  statistics regularizer that helps on YOLO/ResNet targets contributes nothing here, and the attack has
  to work from the gradient matching term alone.

## How to Run

1. Download COCO val2017 (about 1 GB, enough for single image experiments):

```bash
./download_coco.sh
```

2. Run the attack:

```bash
python main.py
```

Dependencies beyond the LeakPro `federated` extra: `transformers`, `timm` (DETR's ResNet backbone) and
`pycocotools`.

## Credits

- **Attack implementation inspired by:**
  - Geiping et al., 2020 — [_Inverting Gradients: How easy is it to break privacy in federated learning?_](https://arxiv.org/abs/2003.14053)

- **Model:**
  - Carion et al., 2020 — [_End-to-End Object Detection with Transformers_](https://arxiv.org/abs/2005.12872),
    used through the Apache-2.0 licensed `transformers` implementation and the `facebook/detr-resnet-50`
    checkpoint.
