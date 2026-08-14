#!/bin/bash
# Does freezing the bipartite matching unblock the attack? The client's shared update is bit-identical
# either way (verified exactly), so this isolates the effect of a smooth vs piecewise objective.
cd /home/edgelab/LeakPro/examples/gia/coco_detr_1_image
CROWD="--min-objects 10 --max-objects 100"
COMMON="--iters 1500 --img-size 256 --attack inverting --tv 1e-2 --lr 0.1 --median-pooling 1"

# Stock DETR on a crowded image. Baseline for this cell is obj_10_100 (SSIM 0.0028, loss 0.9236).
python3 run_experiments.py --name fm_detr50_oracle --model detr50 --pretrained 1 --unfreeze-bn 1 \
    --fixed-match 1 $CROWD $COMMON > logs/fm_detr50_oracle.log 2>&1
echo "done fm_detr50_oracle"

# The deployable version: re-solve the matching periodically instead of never.
python3 run_experiments.py --name fm_detr50_refresh --model detr50 --pretrained 1 --unfreeze-bn 1 \
    --fixed-match 1 --refresh-every 50 $CROWD $COMMON > logs/fm_detr50_refresh.log 2>&1
echo "done fm_detr50_refresh"

# The best variant on the ladder, with and without the frozen matching, on the same crowded image.
python3 run_experiments.py --name fm_notrans_base --model small --variant r18-notrans --pretrained 1 \
    $CROWD $COMMON > logs/fm_notrans_base.log 2>&1
echo "done fm_notrans_base"
python3 run_experiments.py --name fm_notrans_oracle --model small --variant r18-notrans --pretrained 1 \
    --fixed-match 1 $CROWD $COMMON > logs/fm_notrans_oracle.log 2>&1
echo "done fm_notrans_oracle"

# And the runner-up, same pair.
python3 run_experiments.py --name fm_s3_base --model small --variant r18-s3 --pretrained 1 \
    $CROWD $COMMON > logs/fm_s3_base.log 2>&1
echo "done fm_s3_base"
python3 run_experiments.py --name fm_s3_oracle --model small --variant r18-s3 --pretrained 1 \
    --fixed-match 1 $CROWD $COMMON > logs/fm_s3_oracle.log 2>&1
echo "done fm_s3_oracle"
echo FMDONE
