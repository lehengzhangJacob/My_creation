#!/usr/bin/env bash
# MOAC-style step-loss run for MOSI paper-row trial 39.
set -euo pipefail

MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

OUT="${MY_CREATION}/loss_curve_analysis/runs/mosi_trial39_step_loss"
CKPT="${OUT}/checkpoints"
STEP_CSV="${OUT}/step_losses.csv"
mkdir -p "$CKPT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "train log: ${OUT}/train.log"
echo "step csv : ${STEP_CSV}"
echo "checkpoints: ${CKPT}"
"$PYTHON" -u ablation_study/train_fixed_mosi_more_trial39.py \
  --ablation none \
  --checkpoint-dir "$CKPT" \
  --step-loss-csv "$STEP_CSV" \
  --step-loss-log-interval 1 \
  2>&1 | tee "${OUT}/train.log"
