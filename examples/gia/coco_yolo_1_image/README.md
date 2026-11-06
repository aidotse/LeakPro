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

### Modular framework

`yolo_attack.py` attacks the YOLO model of this folder through LeakPro's modular `detection` preset. It runs on a synthetic image (no COCO download) with a FedAvg client, and recovers the image only: `ComputeLoss` gives the box coordinates no gradient, so the true boxes are passed to the attack and only image metrics are reported. Run it from the repository root:

```bash
python -m examples.gia.coco_yolo_1_image.yolo_attack --quick           # short CPU smoke run
python -m examples.gia.coco_yolo_1_image.yolo_attack --device cuda     # full run
```

The tests of this folder (`tests/`) run with `pytest examples/gia/coco_yolo_1_image/tests`.

### Legacy script (COCO)

`main.py` predates the modular framework and uses the legacy attack classes on COCO val2017:

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

The model code in this folder is derived from [jahongir7174/YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt), a PyTorch re-implementation that references Ultralytics' YOLO. The folder carries the GNU Affero General Public License v3 (the Ultralytics code it ultimately derives from is AGPL-3.0); see the [LICENSE](./LICENSE) file next to this README. Nothing in the `leakpro` package, its tests included, imports from this folder: the tests that exercise this model live in [`tests/`](./tests) under the same licence and are not part of the installed package.
