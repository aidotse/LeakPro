#!/usr/bin/env bash
set -euo pipefail

# Adjust these
PYTHON="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-measuremodel.py}"   # <-- change to your filename
GPU="${GPU:-0}"

# Experiment values to run in parallel
EXPS=(0.1)

# Optional knobs
SEED=1234
START_IDX=19
N_TRIALS=10
AT_ITER=10000
NUM_WORKERS=2

mkdir -p logs

echo "Launching ${#EXPS[@]} runs on GPU ${GPU}..."
echo "Logs -> ./logs/"

for exp in "${EXPS[@]}"; do
  log="logs/exp_${exp}_$(date +%Y%m%d_%H%M%S).log"

  echo "  exp=${exp} -> ${log}"
  CUDA_VISIBLE_DEVICES="${GPU}" nohup "${PYTHON}" "${SCRIPT}" \
    --exp "${exp}" \
    --seed "${SEED}" \
    --start-idx "${START_IDX}" \
    --n-trials "${N_TRIALS}" \
    --at-iterations "${AT_ITER}" \
    --num-workers "${NUM_WORKERS}" \
    > "${log}" 2>&1 &
done

echo "All jobs started."
echo "Check:  tail -f logs/exp_0.70_*.log"
echo "Or:     ps aux | grep ${SCRIPT}"

