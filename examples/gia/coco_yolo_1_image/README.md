# Gradient Inversion Attack on Modified YOLO (COCO)

This repository demonstrates an example of a gradient inversion attack, adapted from **Geiping et al., 2020: _"Inverting Gradients - How Easy Is It to Break Privacy in Federated Learning?"_**.  
We apply this attack to a YOLO model trained on the COCO dataset, using a restructured architecture to allow for successful information leakage.

## Key Modifications

Standard YOLO models with CSP (Cross Stage Partial) layers tend to resist gradient inversion due to their depth and skip connections.  
To make the attack feasible, we:

- **Replaced CSP layers** with standard **ResNet BasicBlock** layers.
- **Reduced model depth**, creating a shallower backbone while preserving detection performance on COCO.
- Used a YOLO-style model restructured to align with the COCO dataset format.

These changes allow meaningful visual reconstructions from shared gradients, revealing potential privacy risks in vision models used in federated learning.

## How to Run

1. Clone this repository and ensure dependencies are installed.
2. Download the COCO dataset (val2017) by running:

```bash
   ./download_coco.sh
```
3. Run the main attack script:
```bash
    python main.py
```
This will launch the gradient inversion procedure against the modified YOLO model.

## Credits

- **Attack implementation inspired by:**
  - Geiping et al., 2020 — [_Inverting Gradients: How easy is it to break privacy in federated learning?_](https://arxiv.org/abs/2003.14053)

- **YOLO model base and COCO integration adapted from:**
  - https://github.com/jahongir7174/YOLOv8-pt  
    (A PyTorch re-implementation referencing the original [Ultralytics YOLO](https://github.com/ultralytics/yolov5))

## License

This example builds on Ultralytics' work and follows the **Ultralytics License** terms.  
See the [LICENSE](https://github.com/ultralytics/ultralytics?tab=AGPL-3.0-1-ov-file) file in the original Ultralytics repository for details.
