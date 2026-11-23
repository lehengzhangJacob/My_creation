#!/usr/bin/env bash
# Wait for free VRAM, retrain MOSI paper trial39 with per-step losses,
# aggregate every 10 iters, redraw MOAC panels, copy into Overleaf graph/.
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/home/gmn/.conda/envs/ITHP5090/bin/python}"
NEED_MIB="${NEED_MIB:-11000}"
POLL_SEC="${POLL_SEC:-60}"
OUT="${MY_CREATION}/loss_curve_analysis/runs/mosi_trial39_step_loss"
CKPT="${OUT}/checkpoints"
STEP_CSV="${OUT}/step_losses.csv"
AGG_CSV="${MY_CREATION}/loss_curve_analysis/results/mosi_trial39_stepavg10_losses.csv"
PLOT_DIR="${MY_CREATION}/loss_curve_analysis/results/mosi_trial39_moac_standard"
LOG="${OUT}/wait_rerun.log"
mkdir -p "$CKPT" "$PLOT_DIR" "$(dirname "$AGG_CSV")"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TQDM_DISABLE=1

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', *' -v need="$NEED_MIB" '$2+0>=need {print $1, $2+0}' \
    | sort -k2 -nr \
    | awk 'NR==1{print $1}'
}

log "start NEED_MIB=${NEED_MIB}"
gpu=""
while true; do
  gpu=$(pick_gpu || true)
  if [[ -n "${gpu}" ]]; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    log "GPU${gpu} free=${free}MiB — launching MOSI trial39 step-loss train"
    break
  fi
  freelist=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' ')
  log "waiting; free=[${freelist}]"
  sleep "$POLL_SEC"
done

# Fresh CSV / checkpoints for a clean recalculation
rm -f "$STEP_CSV"
rm -rf "${CKPT:?}/"*

CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u ablation_study/train_fixed_mosi_more_trial39.py \
  --ablation none \
  --checkpoint-dir "$CKPT" \
  --step-loss-csv "$STEP_CSV" \
  --step-loss-log-interval 1 \
  >"${OUT}/train.log" 2>&1
log "train finished; aggregating"

"$PYTHON" loss_curve_analysis/aggregate_step_losses.py \
  --input "$STEP_CSV" \
  --output "$AGG_CSV" \
  --window 10
log "aggregated → ${AGG_CSV}"

"$PYTHON" loss_curve_analysis/plot_moac_standard_panels.py \
  --csv "$AGG_CSV" \
  --out-dir "$PLOT_DIR"
log "plotted → ${PLOT_DIR}"

# Copy the five paper panels into manuscript graph dirs (keep tex include names)
for dst in \
  "${MY_CREATION}/../overleaf_InfoFusion/graph" \
  "${MY_CREATION}/../Information_Fusion_manuscript/graph"
do
  [[ -d "$dst" ]] || continue
  for stem in total_loss task_loss token_ib label_ib routing; do
    cp -f "${PLOT_DIR}/mosi_trial39_moac_${stem}.pdf" "${dst}/mosi_trial39_moac_${stem}.pdf"
  done
  log "copied panels → ${dst}"
done

log "ALL DONE"
