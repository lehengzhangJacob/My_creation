#!/usr/bin/env bash
# Replay frozen Nov-2026 trials in the background (survives SSH disconnect).
#
#   bash fixed_experiment/run_sota_2026_11.sh mustard_trial1239            # GPU 0
#   GPU=2 bash fixed_experiment/run_sota_2026_11.sh simsv2_trial159
#   GPU=3 bash fixed_experiment/run_sota_2026_11.sh mosei_deberta_trial36
#   GPU=1 ABLATION=no_ib bash fixed_experiment/run_sota_2026_11.sh simsv2_trial159   # w/o VTB
#
# Outputs: fixed_experiment/runs/sota_2026_11/<trial>[_<ablation>]/{train.log,checkpoints/,check.log}
set -euo pipefail
TRIAL="${1:?usage: run_sota_2026_11.sh <simsv2_trial159|mustard_trial1239|mosei_deberta_trial36>}"
ABLATION="${ABLATION:-none}"
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/home/gmn/.conda/envs/prism/bin/python}"
RUN="$TRIAL"; [[ "$ABLATION" != none ]] && RUN="${TRIAL}_${ABLATION}"
OUT="${MY_CREATION}/fixed_experiment/runs/sota_2026_11/${RUN}"
mkdir -p "$OUT"

for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset "$v"; done
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1

nohup "$PYTHON" -u fixed_experiment/train_fixed_sota_2026_11.py --trial "$TRIAL" --ablation "$ABLATION" \
  --out-dir "$OUT" > "${OUT}/check.log" 2>&1 &
echo "started ${RUN} on GPU ${CUDA_VISIBLE_DEVICES} pid $!"
echo "  train log : ${OUT}/train.log"
echo "  result    : ${OUT}/check.log"
