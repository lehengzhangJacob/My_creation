#!/usr/bin/env bash
# Aggregate step losses and plot MOAC-style curves (no training).
set -euo pipefail

MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import sys" 2>/dev/null; then
  PYTHON=python3
fi

STEP_CSV="${MY_CREATION}/loss_curve_analysis/runs/mosei_trial70_step_loss/step_losses.csv"
AGG_CSV="${MY_CREATION}/loss_curve_analysis/results/mosei_trial70_stepavg10_losses.csv"
OUT_PDF="${MY_CREATION}/loss_curve_analysis/results/mosei_trial70_moac_stepavg10_losses.pdf"
OUT_NO_ROUTING_PDF="${MY_CREATION}/loss_curve_analysis/results/mosei_trial70_moac_stepavg10_losses_no_routing.pdf"

if [[ ! -s "$STEP_CSV" ]]; then
  echo "Missing or empty: $STEP_CSV" >&2
  exit 1
fi

"$PYTHON" loss_curve_analysis/aggregate_step_losses.py \
  --input "$STEP_CSV" \
  --output "$AGG_CSV" \
  --window 10

"$PYTHON" loss_curve_analysis/plot_loss_curves.py \
  --moac-step \
  --inputs "$AGG_CSV" \
  --output "$OUT_PDF" \
  --also-png

"$PYTHON" loss_curve_analysis/plot_loss_curves.py \
  --moac-step \
  --hide-routing \
  --inputs "$AGG_CSV" \
  --output "$OUT_NO_ROUTING_PDF" \
  --also-png

echo "Done: $AGG_CSV"
echo "Done: ${OUT_PDF%.pdf}.png"
echo "Done: ${OUT_NO_ROUTING_PDF%.pdf}.png"
