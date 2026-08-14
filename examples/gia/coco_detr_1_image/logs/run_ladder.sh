#!/bin/bash
# Step 3, first pass: walk the architecture ladder at fixed hyperparameters, so the numbers are
# directly comparable to the existing stock-DETR table. Cheap; picks which variants deserve a search.
cd /home/edgelab/LeakPro/examples/gia/coco_detr_1_image
for v in r18 r18-s3 r18-shallow r18-shallow-s3 r18-nodec r18-notrans; do
  python3 run_experiments.py --name "ladder_${v}" --model small --variant "$v" --pretrained 1 \
      --iters 1500 --img-size 256 --attack inverting --tv 1e-2 --lr 0.1 --median-pooling 1 \
      > "logs/ladder_${v}.log" 2>&1
  echo "done $v"
done
echo LADDERDONE
