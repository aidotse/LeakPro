#!/bin/bash
# Step 2: does the plateau go away at lower resolution?
cd /home/edgelab/LeakPro/examples/gia/coco_detr_1_image
for size in 64 128; do
  python3 run_experiments.py --name "res${size}_inv" --iters 1500 --img-size $size \
      --pretrained 1 --unfreeze-bn 1 --attack inverting --tv 1e-2 --median-pooling 0 \
      > logs/res${size}_inv.log 2>&1
done
python3 run_experiments.py --name "res64_base" --iters 1500 --img-size 64 \
    --pretrained 1 --unfreeze-bn 1 --attack base --tv 1e-2 --median-pooling 0 \
    > logs/res64_base.log 2>&1
echo ALLDONE
