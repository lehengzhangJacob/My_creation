# Reviewer experiment harness

This directory implements the plan's step 2 without duplicating frozen
hyperparameters. Every new run calls an existing
`ablation_study/train_fixed_*.py` launcher and passes only the reviewer-control
override plus a unique checkpoint directory.

## Matrix

`manifest.json` contains 33 experiment records:

- 18 runnable seed-128 runs: 2 text-only DeBERTa, 4 confidence, 8 routing,
  and 4 paired HCF/no-HCF runs.
- 2 MOSI/MOSEI PRISM+BERT records marked `external_running`. They are managed
  by another orchestrator and are structurally non-runnable here.
- 13 completed controls marked `action: "reuse"`: full, no-confidence,
  fixed-text/no-DPR, no-InfoGate, and the available no-adaptive-gate result.
- Existing MOSI/MOSEI five-seed summaries and MOSEI/SIMSv2 routing CSVs are
  listed separately as reusable assets.

Reuse entries have no launcher or overrides, so the queue cannot retrain them.
The manifest validator also rejects copied training hyperparameters; only
`seed`, backbone/model variant, confidence, routing, routing-supervision, and
HCF overrides are accepted.

## Environment and safety

Run all commands in the project environment:

```bash
conda run -n ITHP5090 python My_creation/review_experiments/launch_review_matrix.py --dry-run
```

The launcher defaults to exactly `CUDA_VISIBLE_DEVICES=0` and rejects a
comma-separated GPU list. A nonblocking file lock also prevents two real
launcher instances from training concurrently.

MOSI/MOSEI PRISM+BERT entries have no launcher or overrides. Both the default
queue and explicit `--all` print them as `EXTERNAL ... [NEVER LAUNCH]` and skip
them. Even `--only mosi_prism_bert_s128` cannot launch training. Manifest
validation also rejects any future runnable MOSI/MOSEI entry that requests a
BERT text backbone.

## Serial launcher

Inspect subsets without launching training:

```bash
conda run -n ITHP5090 python My_creation/review_experiments/launch_review_matrix.py \
  --dry-run --only 'mosei_router_*'
conda run -n ITHP5090 python My_creation/review_experiments/launch_review_matrix.py \
  --dry-run --only confidence
```

After prerequisites are present, a real resumable queue is:

```bash
conda run -n ITHP5090 python My_creation/review_experiments/launch_review_matrix.py --resume
```

Useful options:

- `--only`: repeatable ID glob or exact dataset/family/tag selector.
- `--all`: explicit full-manifest selection; external-running BERT remains
  non-runnable (the no-selector default behaves identically).
- `--resume`: skip only a success marker whose manifest digest and expected
  checkpoint/prediction files still match.
- `--force`: explicitly rerun a successful entry.
- `--keep-going`: continue serially after a failure; it never adds concurrency.
- `--cuda-visible-devices 0`: choose one GPU; default is `0`.

Each subprocess writes to a temporary log that is flushed and atomically renamed
to `logs/<experiment>.log` on exit. State transitions and
`state/<experiment>.success.json` are atomically replaced. A success marker is
created only after exit code 0 and after both the dev-selected checkpoint and
`best_test_predictions.csv` exist.

For each HCF dataset the queue intentionally contains two fresh seed-128 runs.
`--disable-hcf` is passed only to the no-HCF member. The frozen classification
path then sets `hcf_dim=0`, makes the loader emit four tensors rather than five,
omits the HCF projector/encoder, and constructs a three-way rather than
four-way router.

## Result collection

Collect all available results:

```bash
conda run -n ITHP5090 python My_creation/review_experiments/collect_results.py
```

The collector prefers checkpoint fields in this order of evidence:
`test_results`, `selection_score`, `epoch`, `args`, and paired
`test_preds`/`test_labels`. If no usable checkpoint exists, it parses only the
final `Best Results (...)` block from the log. It deliberately stops before
`Last Epoch` and `Best Test ... (oracle)` sections.

Outputs:

- `results/summary.json`
- `results/summary.csv`
- `results/predictions/<experiment>/best_test_predictions.csv` when checkpoint
  prediction arrays are available

Metric values in summaries are fractions (for example, `0.874`, not `87.4`).
Every row states that the headline is the test result from the
development-selected checkpoint and sets `test_oracle=false`. Use `--strict`
when missing experiments should make collection return nonzero.

Import an externally completed BERT run without editing the manifest:

```bash
conda run -n ITHP5090 python My_creation/review_experiments/collect_results.py \
  --only mosi_prism_bert_s128 \
  --checkpoint mosi_prism_bert_s128=/path/to/run-or-checkpoint \
  --predictions mosi_prism_bert_s128=/path/to/best_test_predictions.csv
```

`--checkpoint` and its alias `--artifact` accept either a `.pt` file or a run
directory containing the expected checkpoint. Repeatable `--log` and
`--predictions` imports are also available. These options are collector-only
and never make an external entry runnable.

## Paired bootstrap

Both inputs must be per-sample `best_test_predictions.csv` files for the same
test examples in exactly the same order:

```bash
conda run -n ITHP5090 python My_creation/review_experiments/paired_bootstrap.py \
  --a path/to/control/best_test_predictions.csv \
  --b path/to/full/best_test_predictions.csv \
  --task regression --metric mae --seed 128 --n-bootstrap 10000 \
  --output My_creation/review_experiments/results/control_vs_full.json
```

Binary tasks support `accuracy` and `f1`; regression supports `mae` and `rmse`.
The output reports `metric(a) - metric(b)`, a percentile 95% confidence
interval, and a two-sided paired-bootstrap p-value. Length, sample IDs, and
labels are checked exactly before resampling. Missing files, aggregate-only
tables, and external baselines without per-sample predictions are rejected.

## Lightweight checks

These checks do not start training:

```bash
conda run -n ITHP5090 python -m unittest discover \
  -s My_creation/review_experiments/tests -v
conda run -n ITHP5090 python My_creation/review_experiments/launch_review_matrix.py \
  --dry-run --only 'mosei_router_argmax_s128'
conda run -n ITHP5090 python My_creation/review_experiments/collect_results.py \
  --only existing_control
```
