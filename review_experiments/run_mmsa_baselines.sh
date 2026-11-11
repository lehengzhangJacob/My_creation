#!/usr/bin/env bash
# Run inside conda env mmsa, on a GPU instance.
#   conda activate mmsa
#   export TMPDIR=/root/autodl-tmp/tmp
# SIMS v2 example:
#   bash run_mmsa_baselines.sh --dataset simsv2 --model mult --seeds 128,42,2024
# MOSEI example:
#   bash run_mmsa_baselines.sh --dataset mosei --model misa --seeds 128
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}/My_creation/review_experiments:${PYTHONPATH:-}"
python "${ROOT}/My_creation/review_experiments/run_mmsa_baselines.py" "$@"
