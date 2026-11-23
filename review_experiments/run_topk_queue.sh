#!/usr/bin/env bash
# Top-2 hard routing ablations on GPU 0/1 only. Waits for free memory.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/gmn/.conda/envs/prism/bin/python}"
MAX_JOBS="${MAX_JOBS:-1}"
LOG_DIR="review_experiments/runs/_topk_queue"
mkdir -p "$LOG_DIR"
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset "$v" || true; done

# name launcher checkpoint_dir need_mib
JOBS=(
  "mosei_router_topk_s128|ablation_study/train_fixed_mosei_phase1_trial70.py|review_experiments/runs/mosei_router_topk_s128/checkpoints|9000"
  "simsv2_router_topk_s128|ablation_study/train_fixed_simsv2_tier3_fresh_trial53.py|review_experiments/runs/simsv2_router_topk_s128/checkpoints|7000"
)

declare -A PID_OF=()

is_done() {
  local ckpt="$1"
  local log
  log="$(dirname "$ckpt")/../train.log"
  # train.py writes next to the process stdout we capture, and Best Results in that log
  local jlog="$2"
  [[ -f "$jlog" ]] && grep -q "Best Results" "$jlog"
}

pick_gpu() {
  local need="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', *' -v need="$need" '
      {
        idx=$1+0; free=$2+0;
        if (idx!=0 && idx!=1) next;
        if (free>=need) { print idx; exit }
      }'
}

echo "$(date -Is) topk queue start MAX_JOBS=$MAX_JOBS" | tee -a "$LOG_DIR/queue.log"

while true; do
  for key in "${!PID_OF[@]}"; do
    pid="${PID_OF[$key]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      echo "$(date -Is) finished $key pid=$pid" | tee -a "$LOG_DIR/queue.log"
      unset "PID_OF[$key]"
    fi
  done
  left=0
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r name launcher ckpt need <<<"$spec"
    jlog="$LOG_DIR/${name}.log"
    [[ -n "${PID_OF[$name]+x}" ]] && continue
    if is_done "$ckpt" "$jlog"; then
      continue
    fi
    left=$((left+1))
  done
  if (( left==0 )) && (( ${#PID_OF[@]}==0 )); then
    echo "$(date -Is) all topk jobs done" | tee -a "$LOG_DIR/queue.log"
    break
  fi
  while (( ${#PID_OF[@]} < MAX_JOBS )); do
    launched=0
    for spec in "${JOBS[@]}"; do
      IFS='|' read -r name launcher ckpt need <<<"$spec"
      jlog="$LOG_DIR/${name}.log"
      [[ -n "${PID_OF[$name]+x}" ]] && continue
      if is_done "$ckpt" "$jlog"; then
        continue
      fi
      gpu="$(pick_gpu "$need" || true)"
      if [[ -z "${gpu:-}" ]]; then
        continue
      fi
      echo "$(date -Is) launch $name on GPU${gpu} need=${need}" | tee -a "$LOG_DIR/queue.log"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$PYTHON" -u "$launcher" --routing-mode topk --seed 128 --checkpoint-dir "$ckpt"
      ) >"$jlog" 2>&1 &
      PID_OF[$name]=$!
      launched=1
      sleep 15
      break
    done
    if (( launched==0 )); then
      break
    fi
  done
  sleep 30
done
