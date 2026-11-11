#!/usr/bin/env bash
# Serial full-training complexity timing: MOSI then MOSEI (one GPU, no interference).
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "${MY_CREATION}"
export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_COMPLEXITY_TMPDIR="${MODEL_COMPLEXITY_TMPDIR:-${MY_CREATION}/Model_Complexity_Analysis/tmp}"
export TMPDIR="${TMPDIR:-${MODEL_COMPLEXITY_TMPDIR}}"
mkdir -p "${MODEL_COMPLEXITY_TMPDIR}"
LOG="${LOG:-Model_Complexity_Analysis/nohup_fulltrain_serial_20260525.log}"
DATE_TAG="${DATE_TAG:-20260525}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

log "=== full-training complexity (serial) GPU=${CUDA_VISIBLE_DEVICES} ==="
log "temp dir: ${MODEL_COMPLEXITY_TMPDIR}"

log "--- MOSI (trial39 paper config) ---"
./Model_Complexity_Analysis/run_measure.sh \
  --cases mosi \
  --full-training-time \
  --output "Model_Complexity_Analysis/results_mosi_fulltrain_paper_${DATE_TAG}.md" \
  2>&1 | tee -a "${LOG}"

log "--- MOSEI (phase1 trial70 paper config) ---"
./Model_Complexity_Analysis/run_measure.sh \
  --cases mosei \
  --full-training-time \
  --output "Model_Complexity_Analysis/results_mosei_fulltrain_paper_${DATE_TAG}.md" \
  2>&1 | tee -a "${LOG}"

log "=== all done ==="
