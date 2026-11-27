#!/usr/bin/env bash
# MOSEI DeBERTa trial 36 step-loss curves. Waits for a free GPU, then trains,
# plots, and updates the Overleaf figure. Detached from the launching shell.
set -uo pipefail
MY="/home/gmn/danger/codes/PRISM2/My_creation"
PY="/home/gmn/.conda/envs/prism/bin/python"
OUT="${MY}/loss_curve_analysis/runs/mosei_trial36_step_loss"
CKPT="${OUT}/checkpoints"
CSV="${OUT}/step_losses.csv"
AVG="${MY}/loss_curve_analysis/results/mosei_trial36_stepavg10_losses.csv"
PANEL="${MY}/loss_curve_analysis/results/mosei_trial36_moac_standard"
GRAPH="/home/gmn/danger/codes/PRISM2/overleaf_InfoFusion/graph"
NEED=10500
mkdir -p "$CKPT" "$PANEL"
cd "$MY"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset "$v" || true; done

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v need="$NEED" '$2+0 >= need { print $1; exit }'
}

echo "$(date -Is) waiting for a GPU with >= ${NEED} MiB free"
GPU=""
while [[ -z "$GPU" ]]; do
  GPU="$(pick_gpu || true)"
  if [[ -z "$GPU" ]]; then
    sleep 120
  fi
done
echo "$(date -Is) launching on GPU ${GPU}"
export CUDA_VISIBLE_DEVICES="$GPU"

CMD_FILE="${OUT}/cmd.txt"
"$PY" - "$CMD_FILE" "$CKPT" "$CSV" << 'PY'
import json, sys
cmd_file, ckpt, csv = sys.argv[1:]
src = "/home/gmn/danger/codes/PRISM2/My_creation/sota_search/runs/mosei_deberta/trial_36/cmd.json"
cmd = json.load(open(src))["cmd"]
out = []
skip = False
for arg in cmd:
    if skip:
        out.append(ckpt)
        skip = False
        continue
    if arg == "--checkpoint_dir":
        out.append(arg)
        skip = True
    else:
        out.append(arg)
out[0] = "/home/gmn/.conda/envs/prism/bin/python"
out += ["--step_loss_csv", csv, "--step_loss_log_interval", "1"]
open(cmd_file, "w").write("\n".join(out) + "\n")
PY
mapfile -t CMD < "$CMD_FILE"
"${CMD[@]}" > "${OUT}/train.log" 2>&1
rc=$?
echo "$(date -Is) train exit rc=${rc}"
if [[ $rc -ne 0 ]]; then
  exit "$rc"
fi

"$PY" -u loss_curve_analysis/aggregate_step_losses.py \
  --input "$CSV" --output "$AVG" --window 10
"$PY" -u loss_curve_analysis/plot_moac_standard_panels.py \
  --csv "$AVG" --out-dir "$PANEL" --prefix mosei_trial36 --no-composite

for stem in total_loss task_loss token_ib label_ib routing; do
  cp -f "${PANEL}/mosei_trial36_moac_${stem}.pdf" "${GRAPH}/mosei_trial36_moac_${stem}.pdf"
done

"$PY" - << 'PY'
from pathlib import Path
old_inc = [
    "mosi_trial39_moac_total_loss.pdf",
    "mosi_trial39_moac_task_loss.pdf",
    "mosi_trial39_moac_token_ib.pdf",
    "mosi_trial39_moac_label_ib.pdf",
    "mosi_trial39_moac_routing.pdf",
]
new_inc = [name.replace("mosi_trial39", "mosei_trial36") for name in old_inc]
for path in (
    Path("/home/gmn/danger/codes/PRISM2/overleaf_InfoFusion/cas-dc-template.tex"),
    Path("/home/gmn/danger/codes/PRISM2/overleaf_InfoFusion/cas-dc-sample.tex"),
):
    text = path.read_text()
    text = text.replace(
        "Figure~\\ref{fig:mosi_loss_curves} shows CMU-MOSI learning curves",
        "Figure~\\ref{fig:mosei_loss_curves} shows CMU-MOSEI learning curves",
    )
    text = text.replace("fig:mosi_loss_curves", "fig:mosei_loss_curves")
    text = text.replace(
        "Learning curves of PRISM loss components on CMU-MOSI.",
        "Learning curves of PRISM loss components on CMU-MOSEI.",
    )
    for src, dst in zip(old_inc, new_inc):
        text = text.replace(src, dst)
    path.write_text(text)
    print("updated", path)
PY

cd /home/gmn/danger/codes/PRISM2/overleaf_InfoFusion
git add cas-dc-template.tex cas-dc-sample.tex \
  graph/mosei_trial36_moac_total_loss.pdf \
  graph/mosei_trial36_moac_task_loss.pdf \
  graph/mosei_trial36_moac_token_ib.pdf \
  graph/mosei_trial36_moac_label_ib.pdf \
  graph/mosei_trial36_moac_routing.pdf
GIT_AUTHOR_NAME='Jacobyakobc' GIT_AUTHOR_EMAIL='jacobyakobc@users.noreply.github.com' \
GIT_COMMITTER_NAME='Jacobyakobc' GIT_COMMITTER_EMAIL='jacobyakobc@users.noreply.github.com' \
git commit -m "Replace the MOSI loss curves with the MOSEI trial 36 run from this machine."
git -c http.proxy=http://127.0.0.1:7890 -c https.proxy=http://127.0.0.1:7890 \
  -c credential.helper='store --file /home/gmn/.git-credentials-overleaf' push origin HEAD
echo "$(date -Is) LOSS_CURVE_DONE"
