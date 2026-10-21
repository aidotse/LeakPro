#!/usr/bin/env bash
# Run reinit_imagenet_attack.py for all five ImageNet models in a tmux session "reinit", one window per model,
# spread over the GPUs.
# Usage (from anywhere):  ./run_reinit_attack.sh        then  tmux attach -t reinit  (Ctrl-b n: next window)
#   GPUS="0 1 2 3" N_TRIALS=30 PYTHON=python3 MODELS="swin_t swin_v2_t" ./run_reinit_attack.sh
# Logs go to reinit_logs/<model>.log, results to leakpro_output/results/reinit_<model>/.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
N_TRIALS="${N_TRIALS:-30}"
SESSION="${SESSION:-reinit}"
# with 4 GPUs the 5th model shares GPU 0, so the two lightest (Swin-T, SwinV2-T) are first and last
MODELS=(${MODELS:-swin_t vit_b_16 convnext_t maxvit_t swin_v2_t})
GPUS=(${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')})

# download/extract Imagenette once before the parallel runs, so they don't race on it
"${PYTHON}" -c "import reinit_imagenet_attack as r; r.get_imagenette_loader(start_idx=r.TARGET_IDX, num_workers=0)"

mkdir -p reinit_logs
tmux new-session -d -s "${SESSION}" -n main -c "${PWD}"
for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  echo "${model} -> GPU ${gpu}, tmux window ${SESSION}:${model}, reinit_logs/${model}.log"
  tmux new-window -d -t "${SESSION}" -n "${model}" -c "${PWD}" \
    "CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} reinit_imagenet_attack.py --model ${model} --n-trials ${N_TRIALS} \
     2>&1 | tee reinit_logs/${model}.log; echo '${model} finished'; exec bash"
done
echo "Started. Attach: tmux attach -t ${SESSION}   Progress: grep -h 'final sim' reinit_logs/*.log | wc -l"
