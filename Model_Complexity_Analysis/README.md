# Model complexity analysis (paper configs)

Case study for the frozen Optuna configurations used by the current paper tables:

| Case | Hyperparameters module | Backbone |
|------|------------------------|----------|
| MOSI more/mosi trial 39 | [`../ablation_study/mosi_more_trial39_hparams.py`](../ablation_study/mosi_more_trial39_hparams.py) | DeBERTa-v3 |
| MOSEI phase1 trial 70 | [`../ablation_study/mosei_phase1_trial70_hparams.py`](../ablation_study/mosei_phase1_trial70_hparams.py) | DeBERTa-v3 |
| SIMSv2 phase4 trial 52 | [`../fixed_experiment/simsv2_phase4_trial52_hparams.py`](../fixed_experiment/simsv2_phase4_trial52_hparams.py) | BERT-base-Chinese |

## Python environment

Use conda env **`ITHP5090`** (same default as [`run_optuna_4090d_restart.sh`](../run_optuna_4090d_restart.sh): PyTorch, Transformers, Optuna stack). The driver should also be started with **`ITHP5090`** (see `run_measure.sh` or the explicit `python` path below). Per-case **worker** subprocesses default to the same **`ITHP5090`** interpreter inside `measure_fixed_cases.py` unless you set **`MODEL_COMPLEXITY_PYTHON`** to another `python` binary.

## Dependencies

Inside **`ITHP5090`**, install one FLOP counting library (once):

```bash
conda activate ITHP5090
pip install fvcore
# or: pip install thop
```

`fvcore` is preferred; if it is not installed, the script tries `thop`, then omits FLOPs with a short reason.

## How to run

Working directory must allow imports and data paths like the frozen trainers (same as `train_fixed_*.py`).

```bash
cd /path/to/My_creation
# Recommended: wrapper pins ITHP5090 (override with PYTHON=... if needed)
chmod +x Model_Complexity_Analysis/run_measure.sh
./Model_Complexity_Analysis/run_measure.sh --cases mosi,mosei,simsv2 --output Model_Complexity_Analysis/results.md
```

**Full training time (serial, nohup)** — MOSI then MOSEI on one GPU:

```bash
cd /path/to/My_creation
nohup bash Model_Complexity_Analysis/run_fulltrain_serial_nohup.sh \
  > Model_Complexity_Analysis/nohup_fulltrain_serial_20260525.log 2>&1 &
tail -f Model_Complexity_Analysis/nohup_fulltrain_serial_20260525.log
```

Outputs: `results_mosi_fulltrain_paper_20260525.md`, `results_mosei_fulltrain_paper_20260525.md`.

Equivalent explicit call:

```bash
cd /path/to/My_creation
/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python -u Model_Complexity_Analysis/measure_fixed_cases.py \
  --cases mosi --full-training-time --output Model_Complexity_Analysis/results_mosi_fulltrain.md
```

Optional:

- **`MODEL_COMPLEXITY_PYTHON`** — path to `python` for per-case worker subprocesses (defaults to `ITHP5090` inside `measure_fixed_cases.py`).
- `--output PATH` — write a Markdown table (e.g. `Model_Complexity_Analysis/results.md`).
- `--memory-full-accum` — in non-full mode, report peak GPU memory after `gradient_accumulation_step` micro-batch backward passes **without** `optimizer.step()` (closer to worst-case peak within an accumulation window).
- `--skip-epoch-time` — skip full `train_epoch` timing (faster smoke test).
- `--full-training-time` — run the complete training loop once and report full-training metrics: average epoch FLOPs estimate, peak GPU memory across the loop, average complete-epoch time, total wall-clock time, and completed epochs. Run large datasets one case at a time to avoid GPU interference.

## Audit-grade forward-only FLOPs

For a stricter forward-only audit, use `measure_audit_forward_only.py`. It
enumerates the real train/dev/test DataLoaders, counts every forward call from
the completed training run, and averages the summed forward FLOPs over completed
epochs. It caches profiler results only for exactly matching batch signatures
(`split`, `stage`, train/eval mode, and tensor shapes).
The script installs extra `fvcore` handles for transformer/InfoGate elementwise,
activation, softmax, reduction, and routing-helper ops so unsupported operators
are surfaced in the report instead of silently omitted.

```bash
cd /path/to/My_creation
CUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python -u \
  Model_Complexity_Analysis/measure_audit_forward_only.py \
  --cases mosi,mosei \
  --output-json Model_Complexity_Analysis/results_audit_forward_only_20260526.json \
  --output-md Model_Complexity_Analysis/results_audit_forward_only_20260526.md \
  --output-csv Model_Complexity_Analysis/results_audit_forward_only_20260526.csv
```

Current audit outputs:

- `results_audit_forward_only_20260526.json`
- `results_audit_forward_only_20260526.md`
- `results_audit_forward_only_20260526.csv`

The audit report also carries forward the measured complete-epoch time and
training-loop peak memory from the serial full-training logs/results when
available.

## Metric definitions

- **Parameters / trainable** — `numel()` over `model.parameters()` and over parameters with `requires_grad` (same spirit as the training startup logs).
- **FLOPs (default / non-full mode)** — approximate count for **one training forward** at the first real training micro-batch shape, `stage=1`, with real-valued labels matching logits shape. Custom heads, gating, and some Transformer ops are only partially supported by generic counters; treat numbers as **comparable estimates**, not audit-grade totals.
- **FLOPs (`--full-training-time`)** — average estimated FLOPs per completed full epoch. The script profiles representative forward FLOPs once for stage 1 and stage 2 on train/dev/test loaders, then estimates each completed epoch as `3 × train_forward_FLOPs × train_batches + dev_forward_FLOPs × dev_batches + test_forward_FLOPs × test_batches`. The `3×` train multiplier is the common forward + backward approximation (`1×` forward plus roughly `2×` backward). The reported value is the mean over actually completed epochs, so early stopping changes the stage weighting.
- **Peak GPU memory (default / non-full mode)** — `torch.cuda.max_memory_allocated()` after **one** forward+backward at micro-batch size `train_batch_size`, with the same loss as training (`L1 + mse_weight * MSE` on predictions + IB term), including loss scaling by `gradient_accumulation_step` as in `train_epoch`. Does not include optimizer state unless you extend the script; it reflects activations + gradients for that step.
- **Peak GPU memory (`--full-training-time`)** — `torch.cuda.max_memory_allocated()` after resetting peak stats immediately before the full epoch loop and reading the maximum after completion or early stopping. This covers training, dev/test evaluation, EMA apply/restore, best-checkpoint saving, and early-stopping bookkeeping inside the measured loop.
- **One-epoch training time** — wall time for **one full** `train_epoch` over the real training `DataLoader`, `stage=1`, including optimizer steps, with `torch.cuda.synchronize()` before/after timing when CUDA is used. First epoch in real training may use EMA only after `ema_start_epoch`; here EMA is disabled (`ema=None`) for a stable single-epoch readout. This column is skipped automatically when `--full-training-time` is used.
- **Average complete-epoch time** — enabled by `--full-training-time`; wall time averaged across all completed epochs of a full training run. Each epoch includes training, development/test evaluation, EMA apply/restore when active, best-checkpoint saving, and early-stopping bookkeeping. This is the `avg_epoch_sec` column in full-training outputs.
- **Total training time** — enabled by `--full-training-time`; wall time from the start of the full epoch loop until completion or early stopping. It excludes one-time data/model construction before the loop and excludes the static parameter/FLOP/memory probes.
- **Completed epochs** — actual number of epochs used for the total/average timing. MOSI paper trial 39 uses early stopping; MOSEI phase1 trial 70 does not, so it runs the configured 50 epochs unless interrupted.

## Outputs

The driver prints a Markdown table to stdout and, with `--output`, writes the same table to a file. Each dataset runs in a **separate subprocess** so the selected `train.py` parses the correct CLI once per case.
