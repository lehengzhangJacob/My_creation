#!/usr/bin/env bash
# Queue habit-seed re-runs when GPUs have free memory. Does not kill other jobs.
#
# From My_creation:
#   SPECS=mosi_deberta:812,mosei_deberta:36 SEEDS=0,1,2,42,123 \
#     MAX_JOBS=2 bash sota_search/run_habit_seed_queue.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/gmn/.conda/envs/prism/bin/python}"
SPECS="${SPECS:-mosi_deberta:812,mosei_deberta:36}"
SEEDS="${SEEDS:-0,1,2,42,123}"
MAX_JOBS="${MAX_JOBS:-2}"
NEED_FALLBACK_MIB="${NEED_FALLBACK_MIB:-9800}"
ALLOWED_GPUS="${ALLOWED_GPUS:-}"
LOG_DIR="sota_search/runs/_habit_seed_queue"
mkdir -p "$LOG_DIR"
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset "$v" || true; done

mapfile -t LINES < <("$PYTHON" sota_search/run_habit_seed_sweep.py plan --specs "$SPECS" --seeds "$SEEDS" 2>/dev/null | grep -v '^#' || true)
echo "$(date -Is) habit-seed queue start n=${#LINES[@]} specs=$SPECS seeds=$SEEDS MAX_JOBS=$MAX_JOBS" | tee -a "$LOG_DIR/queue.log"

declare -A PID_OF=()

is_done() {
  local row="$1" base="$2" seed="$3"
  local log="sota_search/runs/${row}/habit_seed/base${base}_s${seed}/train.log"
  [[ -f "$log" ]] && grep -q "Best Results" "$log" 2>/dev/null
}

reap() {
  local key pid
  for key in "${!PID_OF[@]}"; do
    pid="${PID_OF[$key]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      echo "$(date -Is) finished $key pid=$pid" | tee -a "$LOG_DIR/queue.log"
      unset "PID_OF[$key]"
    fi
  done
}

pick_gpu() {
  local need="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', *' -v need="$need" -v allow="$ALLOWED_GPUS" '
      BEGIN {
        n = split(allow, a, ",");
        for (i = 1; i <= n; i++) if (a[i] != "") ok[a[i] + 0] = 1;
      }
      {
        idx = $1 + 0; free = $2 + 0;
        if (n > 0 && !(idx in ok)) next;
        if (free >= need) { print idx; exit }
      }'
}

next_job() {
  local line row base seed status mem key
  for line in "${LINES[@]}"; do
    IFS=$'\t' read -r row base seed status mem <<<"$line"
    key="${row}:${base}:${seed}"
    [[ -n "${PID_OF[$key]+x}" ]] && continue
    if is_done "$row" "$base" "$seed"; then
      continue
    fi
    echo "$row $base $seed ${mem:-$NEED_FALLBACK_MIB}"
    return 0
  done
  return 1
}

count_left() {
  local n=0 line row base seed status mem key
  for line in "${LINES[@]}"; do
    IFS=$'\t' read -r row base seed status mem <<<"$line"
    key="${row}:${base}:${seed}"
    [[ -n "${PID_OF[$key]+x}" ]] && continue
    if is_done "$row" "$base" "$seed"; then
      continue
    fi
    n=$((n + 1))
  done
  echo "$n"
}

while true; do
  reap
  left="$(count_left)"
  if (( left == 0 )) && (( ${#PID_OF[@]} == 0 )); then
    echo "$(date -Is) all habit-seed jobs done" | tee -a "$LOG_DIR/queue.log"
    break
  fi
  while (( ${#PID_OF[@]} < MAX_JOBS )); do
    job="$(next_job || true)"
    [[ -n "${job:-}" ]] || break
    read -r row base seed mem <<<"$job"
    need="${mem:-$NEED_FALLBACK_MIB}"
    gpu="$(pick_gpu "$need" || true)"
    if [[ -z "${gpu:-}" ]]; then
      break
    fi
    key="${row}:${base}:${seed}"
    jlog="$LOG_DIR/${row}_base${base}_s${seed}.log"
    echo "$(date -Is) launch $key on GPU${gpu} need=${need}" | tee -a "$LOG_DIR/queue.log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PYTHON" -u sota_search/run_habit_seed_sweep.py run-one \
        --row "$row" --base-trial "$base" --seed "$seed" --python "$PYTHON"
    ) >"$jlog" 2>&1 &
    PID_OF[$key]=$!
    sleep 10
  done
  sleep 30
done

echo "$(date -Is) summary:" | tee -a "$LOG_DIR/queue.log"
"$PYTHON" - <<'PY'
import json
from pathlib import Path
from collections import defaultdict
p = Path("sota_search/runs/habit_seed_results.jsonl")
if not p.is_file():
    print("no results yet")
    raise SystemExit(0)
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
by = defaultdict(list)
for r in rows:
    by[(r["row"], r["base_trial"])].append(r)
for (row, base), rs in sorted(by.items()):
    rs = sorted(rs, key=lambda x: x["seed"])
    print(f"\n{row} base{base}: n={len(rs)} met={sum(1 for x in rs if x.get('met'))}/{len(rs)}")
    for r in rs:
        print(f"  seed={r['seed']:>4} met={r.get('met')} score={r['score']:.3f} {r['results']}")
PY
