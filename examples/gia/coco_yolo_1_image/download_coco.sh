mkdir -p COCO/images/train2017
mkdir -p COCO/images/val2017
mkdir -p COCO/annotations
cd COCO

# Download COCO images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip all
unzip train2017.zip -d images/
unzip val2017.zip -d images/
unzip annotations_trainval2017.zip