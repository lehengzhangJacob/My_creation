#!/usr/bin/env bash
# Queue MOSEI trial-36 sensitivity jobs onto free GPUs without preempting search.
# Skips the frozen center (injected from trial 36 at aggregate time).
#
#   bash sensitivity_analysis/run_mosei_trial36_sensitivity_queue.sh
#
# Env:
#   PYTHON      default: prism conda
#   NEED_MIB    free MiB required to launch (default 10500 for DeBERTa trial36)
#   MAX_JOBS    max concurrent sensitivity jobs (default 2)
#   POLL_SEC    poll interval (default 120)
#   RUNS_ROOT   default: My_creation/runs/sensitivity_mosei/trial36
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"
PYTHON="${PYTHON:-/home/gmn/.conda/envs/prism/bin/python}"
NEED_MIB="${NEED_MIB:-10500}"
MAX_JOBS="${MAX_JOBS:-2}"
POLL_SEC="${POLL_SEC:-120}"
RUNS_ROOT="${RUNS_ROOT:-${MY_CREATION}/runs/sensitivity_mosei/trial36}"
LOG_DIR="${RUNS_ROOT}/_queue"
mkdir -p "$LOG_DIR" "$RUNS_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset "$v" || true; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1

mapfile -t JOBS < <("$PYTHON" sensitivity_analysis/run_mosei_trial36_sensitivity.py queue-plan)
echo "$(date -Is) queue start n_jobs=${#JOBS[@]} NEED_MIB=${NEED_MIB} MAX_JOBS=${MAX_JOBS} RUNS_ROOT=${RUNS_ROOT}" | tee -a "${LOG_DIR}/queue.log"

is_done() {
  local axis="$1" val="$2"
  local tag
  tag=$("$PYTHON" -c "
import sys
sys.path.insert(0, 'sensitivity_analysis')
from run_mosei_trial36_sensitivity import value_tag
print(value_tag('$axis', float('$val')))
")
  local log="${RUNS_ROOT}/${axis}/${tag}/train.log"
  # Skip finished OR in-flight (log present) so queue restarts do not double-launch.
  [[ -f "$log" ]]
}

declare -A PID_OF=()

reap() {
  local axis val pid
  for key in "${!PID_OF[@]}"; do
    pid="${PID_OF[$key]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      echo "$(date -Is) finished ${key} pid=${pid}" | tee -a "${LOG_DIR}/queue.log"
      unset "PID_OF[$key]"
    fi
  done
}

next_job() {
  local line axis val
  for line in "${JOBS[@]}"; do
    axis="${line%%$'\t'*}"
    val="${line#*$'\t'}"
    key="${axis}:${val}"
    [[ -n "${PID_OF[$key]+x}" ]] && continue
    if is_done "$axis" "$val"; then
      continue
    fi
    echo "$axis $val"
    return 0
  done
  return 1
}

# Optional: ALLOWED_GPUS=0,1 restricts placement (comma-separated indices).
ALLOWED_GPUS="${ALLOWED_GPUS:-}"
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', *' -v need="$NEED_MIB" -v allow="$ALLOWED_GPUS" '
        BEGIN {
          n = split(allow, a, ",");
          for (i = 1; i <= n; i++) if (a[i] != "") ok[a[i] + 0] = 1;
        }
        {
          g = $1 + 0; free = $2 + 0;
          if (free < need) next;
          if (n > 0 && !(g in ok)) next;
          print g, free;
        }' \
    | sort -k2 -nr \
    | awk 'NR==1{print $1}'
}

all_done=0
while true; do
  reap
  # count remaining
  left=0
  for line in "${JOBS[@]}"; do
    axis="${line%%$'\t'*}"
    val="${line#*$'\t'}"
    key="${axis}:${val}"
    [[ -n "${PID_OF[$key]+x}" ]] && { left=$((left+1)); continue; }
    if ! is_done "$axis" "$val"; then left=$((left+1)); fi
  done
  if (( left == 0 )); then
    echo "$(date -Is) all sensitivity jobs done" | tee -a "${LOG_DIR}/queue.log"
    all_done=1
    break
  fi
  while (( ${#PID_OF[@]} < MAX_JOBS )); do
    job=$(next_job) || break
    axis="${job%% *}"
    val="${job#* }"
    gpu=$(pick_gpu || true)
    if [[ -z "${gpu}" ]]; then
      break
    fi
    key="${axis}:${val}"
    tag_log="${LOG_DIR}/${axis}_$(echo "$val" | tr '.' 'p').out"
    echo "$(date -Is) launch ${axis}=${val} on GPU${gpu}" | tee -a "${LOG_DIR}/queue.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "$PYTHON" -u sensitivity_analysis/run_mosei_trial36_sensitivity.py train \
      --axis "$axis" --value "$val" --runs-root "$RUNS_ROOT" \
      >"$tag_log" 2>&1 &
    PID_OF[$key]=$!
    sleep 20  # let peak alloc settle before next placement
  done
  sleep "$POLL_SEC"
done

RES="${MY_CREATION}/sensitivity_analysis/results/mosei_trial36"
mkdir -p "$RES"
"$PYTHON" sensitivity_analysis/run_mosei_trial36_sensitivity.py aggregate \
  --runs-root "$RUNS_ROOT" \
  --summary-out "${RES}/summary.csv" \
  --inject-center
echo "$(date -Is) aggregated → ${RES}/summary.csv" | tee -a "${LOG_DIR}/queue.log"

PDF="${RES}/mosei_trial36_sensitivity_emoe_style_fontlarge_wspace018.pdf"
"$PYTHON" sensitivity_analysis/plot_mosei_sensitivity_emoe_style.py \
  --summary-csv "${RES}/summary.csv" \
  --output "$PDF" \
  --strict
echo "$(date -Is) plotted → ${PDF}" | tee -a "${LOG_DIR}/queue.log"

# Refresh Overleaf figure (same include path as cas-dc-*.tex)
OL_GRAPH="${MY_CREATION}/../overleaf_InfoFusion/graph"
if [[ -d "$OL_GRAPH" ]]; then
  cp -f "$PDF" "${OL_GRAPH}/mosei_trial70_sensitivity_emoe_style_fontlarge_wspace018.pdf"
  cp -f "$PDF" "${OL_GRAPH}/mosei_trial36_sensitivity_emoe_style_fontlarge_wspace018.pdf"
  echo "$(date -Is) copied PDF into overleaf_InfoFusion/graph/" | tee -a "${LOG_DIR}/queue.log"
fi
exit 0
