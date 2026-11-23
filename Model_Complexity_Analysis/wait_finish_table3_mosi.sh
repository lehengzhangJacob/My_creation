#!/usr/bin/env bash
# Wait for a GPU with enough free VRAM, then finish MOSI eval-FLOPs (OOM remnant)
# and patch results_table3_2026_11.{json,tex} + Overleaf Table 3.
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"
PYTHON="${PYTHON:-/home/gmn/.conda/envs/ITHP5090/bin/python}"
NEED_MIB="${NEED_MIB:-12000}"
POLL_SEC="${POLL_SEC:-60}"
LOG_DIR="${MY_CREATION}/Model_Complexity_Analysis"
LOG="${LOG_DIR}/wait_finish_table3_mosi.log"
JSON="${LOG_DIR}/results_table3_2026_11.json"
TEX="${LOG_DIR}/results_table3_2026_11.tex"
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MODEL_COMPLEXITY_TMPDIR="${MODEL_COMPLEXITY_TMPDIR:-${LOG_DIR}/tmp}"
mkdir -p "$MODEL_COMPLEXITY_TMPDIR"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', *' -v need="$NEED_MIB" '$2+0>=need {print $1, $2+0}' \
    | sort -k2 -nr \
    | awk 'NR==1{print $1}'
}

log "start NEED_MIB=${NEED_MIB} POLL_SEC=${POLL_SEC}"

gpu=""
while true; do
  gpu=$(pick_gpu || true)
  if [[ -n "${gpu}" ]]; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    log "GPU${gpu} free=${free}MiB — launching MOSI remount"
    break
  fi
  freelist=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' ')
  log "waiting; free=[${freelist}]"
  sleep "$POLL_SEC"
done

CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u Model_Complexity_Analysis/measure_table3_paper.py \
  --cases mosi \
  --output-json "${LOG_DIR}/results_table3_mosi_retry.json" \
  --output-tex "${LOG_DIR}/results_table3_mosi_retry.tex" \
  >>"$LOG" 2>&1

"$PYTHON" - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path

root = Path("/home/gmn/danger/codes/PRISM2/My_creation/Model_Complexity_Analysis")
full = json.loads((root / "results_table3_2026_11.json").read_text())
retry = json.loads((root / "results_table3_mosi_retry.json").read_text())
mosi_new = next(r for r in retry if r["case"] == "mosi")
if mosi_new.get("flops_dev_epoch") is None or mosi_new.get("flops_test_epoch") is None:
    raise SystemExit(f"MOSI retry still missing eval FLOPs: {mosi_new}")

by = {r["case"]: r for r in full}
by["mosi"] = mosi_new
order = ["mosi", "mosei", "simsv2"]
rows = [by[c] for c in order if c in by]
(root / "results_table3_2026_11.json").write_text(json.dumps(rows, indent=2) + "\n")

def fmt_params(n):
    return f"{n / 1e6:.1f}M"

def fmt_flops(x):
    if x is None:
        return "n/a"
    ax = abs(x)
    if ax >= 1e12:
        return f"{x / 1e12:.2f}T"
    if ax >= 1e9:
        return f"{x / 1e9:.2f}G"
    if ax >= 1e6:
        return f"{x / 1e6:.2f}M"
    return f"{x:.3g}"

headers = {"mosi": "MOSI", "mosei": "MOSEI", "simsv2": "SIMS-v2"}
head = " & ".join(headers[c["case"]] for c in rows)

def cells(key, fmt):
    return " & ".join(fmt(c[key]) for c in rows)

lines = [
    r"\begin{tabular}{l|" + "c" * len(rows) + "}",
    r"\toprule",
    f"& {head} \\\\",
    r"\midrule",
    f"Parameters & {cells('params', fmt_params)} \\\\",
    f"FLOPs / train fwd & {cells('flops_train_fwd', fmt_flops)} \\\\",
    f"Train fwd FLOPs / epoch & {cells('flops_train_epoch', fmt_flops)} \\\\",
    f"Dev eval FLOPs / epoch & {cells('flops_dev_epoch', fmt_flops)} \\\\",
    f"Test eval FLOPs / epoch & {cells('flops_test_epoch', fmt_flops)} \\\\",
    "Peak train memory (GiB) & "
    + " & ".join(f"{c['peak_mem_gib']:.3f}" for c in rows)
    + r" \\",
    "Avg. epoch time (full loop, s) & "
    + " & ".join(f"{c['epoch_sec']:.2f}" for c in rows)
    + r" \\",
    "Batches / epoch & "
    + " & ".join(f"{c['n_train']:,}".replace(",", "{,}") for c in rows)
    + r" \\",
    "Batch $\\times$ accum. & "
    + " & ".join(f"{c['batch']}$\\times${c['accum']}" for c in rows)
    + r" \\",
    r"\bottomrule",
    r"\end{tabular}",
]
tex = "\n".join(lines) + "\n"
(root / "results_table3_2026_11.tex").write_text(tex)
print(tex)

# Patch Overleaf + local manuscript copies
tabular = tex.strip()
caption = (
    "Model complexity metrics for MOSI, MOSEI, and SIMS-v2. "
    "FLOPs are measured for forward passes only."
)
intro = (
    r"Table~\ref{tab:model_complexity} reports the computational footprint of PRISM on "
    r"CMU-MOSI, CMU-MOSEI, and CH-SIMS~v2 (forward FLOPs only; peak memory and per-epoch "
    r"time on a single NVIDIA GeForce RTX~3090 with the paper micro-batch and "
    r"gradient-accumulation settings)."
)

import re
targets = [
    Path("/home/gmn/danger/codes/PRISM2/overleaf_InfoFusion/cas-dc-sample.tex"),
    Path("/home/gmn/danger/codes/PRISM2/overleaf_InfoFusion/cas-dc-template.tex"),
    Path("/home/gmn/danger/codes/PRISM2/Information_Fusion_manuscript/cas-dc-sample.tex"),
    Path("/home/gmn/danger/codes/PRISM2/Information_Fusion_manuscript/cas-dc-template.tex"),
]
block_pat = re.compile(
    r"(Table~\\ref\{tab:model_complexity\} reports the computational footprint of PRISM on .*?\.)",
    re.S,
)
table_pat = re.compile(
    r"(\\caption\{Model complexity metrics for .*?\}\s*"
    r"\\label\{tab:model_complexity\}\s*)"
    r"(\\begin\{tabular\}.*?\\end\{tabular\})",
    re.S,
)
for path in targets:
    if not path.is_file():
        print("skip missing", path)
        continue
    text = path.read_text()
    text2, n1 = block_pat.subn(intro, text, count=1)
    text3, n2 = table_pat.subn(
        r"\caption{" + caption + r"}\n\label{tab:model_complexity}\n" + tabular,
        text2,
        count=1,
    )
    # Fix caption line if regex above already put full caption in replacement awkwardly
    # Simpler: replace caption + tabular separately
    text = path.read_text()
    text = re.sub(
        r"Table~\\ref\{tab:model_complexity\} reports the computational footprint of PRISM on [^\n]+",
        intro,
        text,
        count=1,
    )
    text = re.sub(
        r"\\caption\{Model complexity metrics for [^}]+\}",
        r"\\caption{" + caption + "}",
        text,
        count=1,
    )
    text = re.sub(
        r"\\begin\{tabular\}\{l\|c+\}.*?\\end\{tabular\}",
        tabular.strip(),
        text,
        count=1,
        flags=re.S,
    )
    path.write_text(text)
    print(f"patched {path}")
print("merged OK")
PY

log "done; restarting sensitivity queue"
cd "$MY_CREATION"
nohup bash sensitivity_analysis/run_mosei_trial36_sensitivity_queue.sh \
  >>"$MY_CREATION/runs/sensitivity_mosei/trial36/_queue/queue.log" 2>&1 &
echo $! >"$MY_CREATION/runs/sensitivity_mosei/trial36/_queue/queue.pid"
log "sensitivity queue pid=$(cat "$MY_CREATION/runs/sensitivity_mosei/trial36/_queue/queue.pid")"
log "ALL DONE"
