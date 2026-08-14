#!/bin/bash
# Does the number of annotated objects drive leakage? Every earlier run attacked COCO image 285,
# which has exactly one object, so only 1 of 100 queries carried positive supervision.
cd /home/edgelab/LeakPro/examples/gia/coco_detr_1_image
for spec in "1 3" "5 8" "10 100"; do
  set -- $spec
  python3 run_experiments.py --name "obj_${1}_${2}" --model detr50 --pretrained 1 --unfreeze-bn 1 \
      --min-objects $1 --max-objects $2 --iters 1500 --img-size 256 \
      --attack inverting --tv 1e-2 --lr 0.1 --median-pooling 1 \
      > "logs/obj_${1}_${2}.log" 2>&1
  echo "done objects $1-$2"
done
echo OBJDONE
