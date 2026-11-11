#!/usr/bin/env bash
# MOAC-style step-loss run for MOSEI phase1 trial 70.
set -euo pipefail

MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

OUT="${MY_CREATION}/loss_curve_analysis/runs/mosei_trial70_step_loss"
CKPT="${OUT}/checkpoints"
STEP_CSV="${OUT}/step_losses.csv"
mkdir -p "$CKPT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "train log: ${OUT}/train.log"
echo "step csv : ${STEP_CSV}"
echo "checkpoints: ${CKPT}"
"$PYTHON" -u ablation_study/train_fixed_mosei_phase1_trial70.py \
  --ablation none \
  --checkpoint-dir "$CKPT" \
  --step-loss-csv "$STEP_CSV" \
  --step-loss-log-interval 1 \
  2>&1 | tee "${OUT}/train.log"
