# Object detection

## detection_attack
Gradient inversion against an object detector: reconstruct the client's images and boxes from one update.

```
python -m examples.gia.object_detection.detection_attack --detector detr|retinanet [--protocol fedsgd|fedavg] [--quick]
```

The script always attacks **random-weight models on synthetic data**: a tiny DETR (built from its config, needs `transformers`) or a RetinaNet-ResNet50 with `weights=None`, and uniformly random 64x64 images that each carry one fixed box. Nothing is downloaded and no COCO data is used. `--quick` only shortens the optimisation (about 10 to 20 s on the CPU); without it the attack runs the full iteration budget, and a GPU is recommended. Because both the model and the images are random, the numbers show that the pipeline works end to end, not how well real detectors leak; the reconstructions are not meaningful. Results go to `examples/gia/outputs/detection_attack/`.

<!-- results:detection_attack -->
| Attack | psnr | ssim | precision_at_iou | recall_at_iou | f1_at_iou | mean_matched_iou | label_accuracy_on_matches | baseline_whole_image_mean_iou | n_true_boxes | n_recon_boxes |
|---|---|---|---|---|---|---|---|---|---|---|
| image | 6.85 ± 0.00 (n=1) | 0.01 ± 0.00 (n=1) |  |  |  |  |  |  |  |  |
| boxes |  |  | 0.00 ± 0.00 (n=1) | 0.00 ± 0.00 (n=1) | 0.00 ± 0.00 (n=1) | 0.00 ± 0.00 (n=1) | 0.00 ± 0.00 (n=1) | 0.22 ± 0.00 (n=1) | 2.00 ± 0.00 (n=1) | 2.00 ± 0.00 (n=1) |

Command: `python -m examples.gia.object_detection.detection_attack --out examples/gia/outputs/detection_attack --detector detr --num-images 2`; torch `2.5.1+cu121`, Quadro RTX 5000.
<!-- /results -->

With the random-weight DETR on random images the reconstruction reaches 6.85 dB PSNR and no true box is matched (precision, recall and F1 at the IoU threshold are 0.00).

For a YOLO-style detector see [`../coco_yolo_1_image`](../coco_yolo_1_image/README.md) (`python -m examples.gia.coco_yolo_1_image.yolo_attack --quick`).
