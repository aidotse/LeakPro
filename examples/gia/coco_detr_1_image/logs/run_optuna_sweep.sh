#!/bin/bash
# Apply the coco_yolo_1_image protocol (optuna over tv, attack_lr, median_pooling, top10norms and
# which image to attack) to DETR. Until now every DETR number came from one hand-picked setting with
# attack_lr fixed at 0.1, while the search range spans 1e-4 to 100.
cd /home/edgelab/LeakPro/examples/gia/coco_detr_1_image
COMMON="--trials 30 --iters 2000 --check-interval 250 --num-trial-images 5 --img-size 256"

# Stock pretrained DETR with trainable BatchNorm, on crowded images.
python3 run_optuna.py --name opt_detr50 --model detr50 --pretrained 1 --unfreeze-bn 1 \
    --attack base --min-objects 10 --max-objects 100 $COMMON > logs/opt_detr50.log 2>&1
echo "done opt_detr50"

# The shallow variant whose gradient profile looked healthiest in layer_analysis.
python3 run_optuna.py --name opt_notrans --model small --variant r18-notrans --pretrained 1 \
    --attack base --min-objects 10 --max-objects 100 $COMMON > logs/opt_notrans.log 2>&1
echo "done opt_notrans"

# Stock-depth small DETR, for contrast with the one above.
python3 run_optuna.py --name opt_r18 --model small --variant r18 --pretrained 1 \
    --attack base --min-objects 10 --max-objects 100 $COMMON > logs/opt_r18.log 2>&1
echo "done opt_r18"
echo OPTUNADONE
