#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""How this section's examples are run: full-size runs for ``common.run_all`` and the cases of the --quick test.

``FULL`` maps a run name (= output directory) to (module, flags of the full run). ``QUICK_VARIANTS`` are extra
--quick cases (module and flags) besides each script with its defaults, and ``REQUIRES`` names the optional packages
a case needs.
"""
OD = "examples.gia.object_detection"
YOLO = "examples.gia.coco_yolo_1_image.yolo_attack"

FULL = {
    "detection_attack": (f"{OD}.detection_attack", ["--detector", "detr", "--num-images", "2"]),
    "detection_attack_retinanet": (f"{OD}.detection_attack", ["--detector", "retinanet", "--num-images", "2"]),
    "yolo_attack": (YOLO, ["--device", "cuda"]),
}
QUICK_VARIANTS = {
    "detection_detr_fedavg": (f"{OD}.detection_attack", "--detector", "detr", "--protocol", "fedavg"),
    "detection_retinanet": (f"{OD}.detection_attack", "--detector", "retinanet"),
    "yolo": (YOLO,),
}
REQUIRES = {"object_detection/detection_attack": ("transformers",),
            "detection_detr_fedavg": ("transformers",), "detection_retinanet": ("transformers",)}
