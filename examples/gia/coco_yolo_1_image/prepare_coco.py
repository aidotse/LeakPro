import os
from glob import glob
from pycocotools.coco import COCO
from tqdm import tqdm

image_dir = "COCO/images/train2017"
output_txt = "COCO/train2017.txt"

# Get all .jpg files, sorted
image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))

with open(output_txt, "w") as f:
    for path in image_paths:
        f.write(path + "\n")

print(f"Saved {len(image_paths)} image paths to {output_txt}")

def coco_to_yolo(json_path, output_folder, image_folder):
    os.makedirs(output_folder, exist_ok=True)
    coco = COCO(json_path)
    image_ids = coco.getImgIds()
    categories = coco.loadCats(coco.getCatIds())
    category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}  # Map to 0-based

    for image_id in tqdm(image_ids):
        img = coco.loadImgs(image_id)[0]
        img_w, img_h = img['width'], img['height']
        file_name = img['file_name']
        image_base = os.path.splitext(file_name)[0]
        label_path = os.path.join(output_folder, f"{image_base}.txt")

        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        lines = []
        for ann in anns:
            if ann.get("iscrowd", 0):  # skip crowd annotations
                continue
            cat_id = category_mapping[ann['category_id']]
            x, y, w, h = ann['bbox']
            x_c = (x + w / 2) / img_w
            y_c = (y + h / 2) / img_h
            w /= img_w
            h /= img_h
            lines.append(f"{cat_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

# Convert both train and val
coco_to_yolo(
    json_path="COCO/annotations/instances_train2017.json",
    output_folder="COCO/labels/train2017",
    image_folder="COCO/images/train2017"
)

coco_to_yolo(
    json_path="COCO/annotations/instances_val2017.json",
    output_folder="COCO/labels/val2017",
    image_folder="COCO/images/val2017"
)