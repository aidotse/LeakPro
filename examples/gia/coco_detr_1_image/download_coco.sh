#!/bin/bash
# Downloads the COCO val2017 split, which is enough for the single image experiments here.
set -e
mkdir -p COCO/images
cd COCO

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -q val2017.zip -d images/
unzip -q annotations_trainval2017.zip

rm val2017.zip annotations_trainval2017.zip
