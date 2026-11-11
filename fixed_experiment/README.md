# Fixed experiment: frozen Optuna trials (MOSI, MOSEI, SIMSv2, UR-FUNNY, MUStARD)

Reproducible single runs with hyperparameters copied from **`My_creation/logs/optuna/4090D_restart`**. CLI float formatting matches `optuna_search_v2.objective()` so parsed learning rates match the gold Optuna train log headers.

**Environment:** same stack as the Optuna driver is recommended (e.g. conda `ITHP5090`, `transformers==4.29.2`). Override interpreter with `PYTHON=/path/to/python` in the shell wrappers.

### MOSEI + SIMSv2 in parallel (two GPUs)

If you have **two** GPUs, bind one job per card to avoid sequential wall-clock:

```bash
cd My_creation && bash fixed_experiment/run_mosei70_simsv52_parallel.sh
```

Defaults: `MOSEI_GPU=0`, `SIMSV2_GPU=1`. Override, for example:

```bash
MOSEI_GPU=1 SIMSV2_GPU=0 bash fixed_experiment/run_mosei70_simsv52_parallel.sh
```

Or start each run yourself with different `CUDA_VISIBLE_DEVICES` in two terminals.

**Note:** Do not chain `run_mosei…; run_simsv2…` in one `bash -c` if you want parallelism; if you already started such a job, you can `kill` the **outer** sequential `bash -c` PID (the parent of `run_mosei_phase1_trial70.sh`)—MOSEI’s `train.py` typically keeps running while the trailing SIMSv2 step is skipped.

## Vendored Python (snapshot)

These files are **copies** of the corresponding modules under [`My_creation/`](../); imports resolve within `fixed_experiment/`. The copied [`train.py`](train.py) patches paths so **`deberta-v3-base`**, **`bert-base-chinese`**, and **`datasets/{mosi,mosei,simsv2}.pkl`** are read from **`My_creation/`**, not from this directory.

| File | Role |
|------|------|
| [`train.py`](train.py) | Entry training script (path patches for snapshot layout) |
| [`deberta_infogate.py`](deberta_infogate.py) | DeBERTa + InfoGate (MOSI / MOSEI) |
| [`bert_infogate.py`](bert_infogate.py) | BERT path (SIMSv2) |
| [`infogate_modules.py`](infogate_modules.py) | Shared InfoGate blocks |
| [`global_configs.py`](global_configs.py) | Dataset dims / device |
| [`simsv2_metrics.py`](simsv2_metrics.py) | SIMSv2 metrics |
| [`selection_utils.py`](selection_utils.py) | Checkpoint selection helpers |

If you change training logic under `My_creation/`, **refresh these copies** when you want this frozen bundle to stay aligned (for example: `cp ../train.py ../deberta_infogate.py ... fixed_experiment/` then re-apply the small path patches in `fixed_experiment/train.py`).

---

## CMU-MOSI — phase4_mosi trial 234

- **Study:** `infogate_mosi_phase4_mosi_4090d`
- **Files:** [`mosi_trial234_hparams.py`](mosi_trial234_hparams.py), [`train_fixed_mosi_trial234.py`](train_fixed_mosi_trial234.py), [`run_mosi_trial234.sh`](run_mosi_trial234.sh)
- **Outputs:** `fixed_experiment/runs/mosi_trial234/` (`train.log`, `checkpoints/`)

```bash
bash /path/to/My_creation/fixed_experiment/run_mosi_trial234.sh
# or: cd My_creation/fixed_experiment && ./run_mosi_trial234.sh
```

Dry-run:

```bash
cd My_creation && python fixed_experiment/train_fixed_mosi_trial234.py --dry-run
```

Verify argv vs gold log:

```bash
cd My_creation && python scripts/verify_mosi_trial234_optuna_train_argv.py
```

Same hyperparameters as [`run_reproduce_mosi_phase4_mosi_trial234.sh`](../run_reproduce_mosi_phase4_mosi_trial234.sh).

---

## CMU-MOSEI — phase1 trial 70

- **Study:** `infogate_mosei_phase1_4090d` (paper / ablation: phase1, trial 70; see `overleaf_69e83a58/acl_latex.tex`)
- **Gold log:** `logs/optuna/4090D_restart/phase1/train_logs/mosei_phase1_trial_70.log`
- **Files:** [`mosei_phase1_trial70_hparams.py`](mosei_phase1_trial70_hparams.py), [`train_fixed_mosei_phase1_trial70.py`](train_fixed_mosei_phase1_trial70.py), [`run_mosei_phase1_trial70.sh`](run_mosei_phase1_trial70.sh)
- **Outputs:** `fixed_experiment/runs/mosei_phase1_trial70/`

Phase1 MOSEI Optuna does **not** pass `--early_stop_patience` (train full `n_epochs`; `train.py` default patience 0).

```bash
bash /path/to/My_creation/fixed_experiment/run_mosei_phase1_trial70.sh
```

Dry-run / verify:

```bash
cd My_creation && python fixed_experiment/train_fixed_mosei_phase1_trial70.py --dry-run
cd My_creation && python scripts/verify_mosei_phase1_trial70_optuna_train_argv.py
```

---

## CH-SIMS v2 — tier3_fresh trial 53 (paper PRISM table row)

- **Study:** `infogate_simsv2_tier3_fresh_mmsa`
- **DB / log:** `logs/optuna/4090D_restart/simsv2_tier3_fresh/db/simsv2.db`, `train_logs/simsv2_tier3_fresh_trial_53.log`
- **Test (Best Results, dev MAE selection):** Acc5 **55.61%**, Acc3 **73.40%**, Acc2 **80.08%**, F1 **80.13%**, MAE **0.2907**, Corr **0.7050** — matches `acl_latex.tex` PRISM row (55.6 / 73.4 / 80.1 / 80.1 / 0.291 / 0.705)
- **Files:** [`simsv2_tier3_fresh_trial53_hparams.py`](simsv2_tier3_fresh_trial53_hparams.py), [`train_fixed_simsv2_tier3_fresh_trial53.py`](train_fixed_simsv2_tier3_fresh_trial53.py), [`run_simsv2_tier3_fresh_trial53.sh`](run_simsv2_tier3_fresh_trial53.sh)
- **Outputs:** `fixed_experiment/runs/simsv2_tier3_fresh_trial53/`

Uses MMSA pkl features (`--simsv2_feature_mode mmsa`), batch 8 × grad accum 4, `--early_stop_patience 15`, `--selection_metric mae`.

```bash
bash /path/to/My_creation/fixed_experiment/run_simsv2_tier3_fresh_trial53.sh
```

**nohup** (SSH 断开后继续跑):

```bash
cd My_creation && bash fixed_experiment/run_simsv2_tier3_fresh_trial53_nohup.sh
# tail -f fixed_experiment/runs/simsv2_tier3_fresh_trial53/train.log
```

Dry-run:

```bash
cd My_creation && python fixed_experiment/train_fixed_simsv2_tier3_fresh_trial53.py --dry-run
```

---

## CH-SIMS v2 — phase4 trial 52

- **Study:** `infogate_simsv2_phase4_4090d` (older phase4 best; **not** the paper PRISM SIMSv2 row — use **tier3_fresh trial 53** above)
- **Files:** [`simsv2_phase4_trial52_hparams.py`](simsv2_phase4_trial52_hparams.py), [`train_fixed_simsv2_phase4_trial52.py`](train_fixed_simsv2_phase4_trial52.py), [`run_simsv2_phase4_trial52.sh`](run_simsv2_phase4_trial52.sh)
- **Outputs:** `fixed_experiment/runs/simsv2_phase4_trial52/`

Uses `--early_stop_patience 15` (same as `run_optuna_4090d_restart.sh` phase4 SIMSv2 default).

```bash
bash /path/to/My_creation/fixed_experiment/run_simsv2_phase4_trial52.sh
```

Dry-run / verify:

```bash
cd My_creation && python fixed_experiment/train_fixed_simsv2_phase4_trial52.py --dry-run
cd My_creation && python scripts/verify_simsv2_phase4_trial52_optuna_train_argv.py
```

---

## UR-FUNNY v2 — trial 162 (paper classification row)

- **Study:** `infogate_ur_funny_optuna_classify_albert_hcf_20260426_192758` (tier-2 TPE)
- **Train log:** `logs/optuna/4090D_restart/classification/optuna_classify_albert_hcf_20260426_192758/train_logs/ur_funny_trial_162.log`
- **Test (Best Results, dev `binary_acc`, epoch ≥ 10):** Acc **75.15%**, F1 **75.01%** — paper **75.2%** (`mhd_msd.py` uses 75.15)
- **Files:** [`ur_funny_trial162_hparams.py`](ur_funny_trial162_hparams.py), [`train_fixed_ur_funny_trial162.py`](train_fixed_ur_funny_trial162.py), [`run_ur_funny_trial162.sh`](run_ur_funny_trial162.sh)
- **Outputs:** `fixed_experiment/runs/ur_funny_trial162/`
- **Entry:** `My_creation/train_classify.py` (ALBERT + HCF, cwd = `My_creation`)

```bash
bash /path/to/My_creation/fixed_experiment/run_ur_funny_trial162.sh
```

Dry-run:

```bash
cd My_creation && python fixed_experiment/train_fixed_ur_funny_trial162.py --dry-run
```

---

## MUStARD — s2_local trial 134 (paper classification row)

- **Study:** `infogate_mustard_optuna_classify_albert_hcf_20260426_192758_s2_local`
- **Train log:** `.../train_logs/mustard_s2_local_trial_134.log`
- **Test (Best Results, dev `binary_acc`, epoch ≥ 13):** Acc **79.41%**, F1 **79.34%** — paper **79.4%**
- **Files:** [`mustard_s2_local_trial134_hparams.py`](mustard_s2_local_trial134_hparams.py), [`train_fixed_mustard_s2_local_trial134.py`](train_fixed_mustard_s2_local_trial134.py), [`run_mustard_s2_local_trial134.sh`](run_mustard_s2_local_trial134.sh)
- **Outputs:** `fixed_experiment/runs/mustard_s2_local_trial134/`

```bash
bash /path/to/My_creation/fixed_experiment/run_mustard_s2_local_trial134.sh
```

**Parallel nohup** (one GPU each; on single-GPU hosts use `UR_FUNNY_GPU=0 MUSTARD_GPU=0`):

```bash
cd My_creation && bash fixed_experiment/run_classify_paper_trials_parallel_nohup.sh
# single GPU example:
UR_FUNNY_GPU=0 MUSTARD_GPU=0 bash fixed_experiment/run_classify_paper_trials_parallel_nohup.sh
```

**Serial / nohup** (UR-FUNNY then MUStARD on one GPU):

```bash
cd My_creation && bash fixed_experiment/run_classify_paper_trials_serial.sh
cd My_creation && bash fixed_experiment/run_classify_paper_trials_nohup.sh
```

Dry-run:

```bash
cd My_creation && python fixed_experiment/train_fixed_mustard_s2_local_trial134.py --dry-run
```

---

## SOTA re-search (Nov 2026) — SIMS-v2 #159, MUStARD #1239, MOSEI DeBERTa #36

Rows that beat every baseline in the 2026-11 Optuna re-search (`sota_search/`). The exact argv of each trial (copied from `sota_search/runs/<row>/trial_<n>/cmd.json`) and its expected test metrics live in [`sota_2026_11_trials.py`](sota_2026_11_trials.py).

| Trial | Expected test metrics | Runtime (3090) |
|---|---|---|
| `simsv2_trial159` | Acc5 57.25 / Acc3 73.50 / Acc2 80.08 / F1 80.15 / MAE 0.2904 / Corr 0.7107 | ~54 min |
| `mustard_trial1239` | Acc 79.41 / F1 79.39 | ~11 min |
| `mosei_deberta_trial36` | Acc7 56.24 / Acc2 87.40 / F1 87.37 / MAE 0.4988 / Corr 0.8035 | ~11.7 h |

```bash
cd My_creation
GPU=1 bash fixed_experiment/run_sota_2026_11.sh mustard_trial1239      # nohup, returns immediately
python fixed_experiment/train_fixed_sota_2026_11.py --trial simsv2_trial159 --dry-run
python fixed_experiment/train_fixed_sota_2026_11.py --trial simsv2_trial159 --check-only   # re-compare a finished run
```

Component ablations reuse the same frozen argv plus one `--ablation` switch (`no_ib` = w/o VTB, `no_mselector` = w/o DPR, `no_infogate` = w/o InfoGate), e.g. `GPU=1 ABLATION=no_ib bash fixed_experiment/run_sota_2026_11.sh simsv2_trial159` → `runs/sota_2026_11/simsv2_trial159_no_ib/`. SIMS-v2 #159 results (paper `tab:ablation`, MAE / Acc5 / Acc2): full 0.290 / 57.3 / 80.1, w/o VTB 0.298 / 56.9 / 79.1, w/o DPR 0.302 / 54.4 / 78.2, w/o InfoGate 0.332 / 52.3 / 78.3.

Output goes to `runs/sota_2026_11/<trial>/` (`train.log`, `check.log`, `checkpoints/`). After training, the launcher compares the final "Best Results" block line by line with the original trial log and prints `EXACT REPRODUCTION` or `MISMATCH` (exit code 0 only on an exact match). Bitwise reproduction is expected only on the same GPU model (RTX 3090) with the `prism` env; other GPUs change dropout masks.

---

## Paper repro bundle

[`scripts/bundle_paper_repro_to_fixed_experiment.sh`](../scripts/bundle_paper_repro_to_fixed_experiment.sh) copies these `*_hparams.py` files into `fixed_experiment/paper_repro_bundle/frozen/` together with thin launchers.
