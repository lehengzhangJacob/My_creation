#!/usr/bin/env bash
# Full 50-epoch MOSEI phase1 trial70 complexity run on GPU1 (real wall time + peak mem).
set -euo pipefail

MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TQDM_DISABLE=1
export MODEL_COMPLEXITY_TMPDIR="${MODEL_COMPLEXITY_TMPDIR:-${MY_CREATION}/Model_Complexity_Analysis/tmp}"
export TMPDIR="${TMPDIR:-${MODEL_COMPLEXITY_TMPDIR}}"
mkdir -p "${MODEL_COMPLEXITY_TMPDIR}"

DATE_TAG="${DATE_TAG:-20260526}"
LOG="${MY_CREATION}/Model_Complexity_Analysis/nohup_mosei_fulltrain_audit_gpu1_${DATE_TAG}.log"
OUT_MD="${MY_CREATION}/Model_Complexity_Analysis/results_mosei_fulltrain_audit_${DATE_TAG}_gpu1.md"

echo "[$(date -Iseconds)] MOSEI full-training-time audit start GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG}"
echo "output: ${OUT_MD}" | tee -a "${LOG}"

./Model_Complexity_Analysis/run_measure.sh \
  --cases mosei \
  --full-training-time \
  --output "${OUT_MD}" \
  2>&1 | tee -a "${LOG}"

echo "[$(date -Iseconds)] MOSEI full-training-time audit done" | tee -a "${LOG}"
