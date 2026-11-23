# Fixed experiments for the current SOTA trials

Frozen replays of the Nov 2026 search trials that met the paper targets. Older fixed experiments (trial 70 / 53 / 234 / 162 / 134, July BERT snapshots, and their logs) are outside this directory, in `PRISM2/fixed_experiment_previous/`.

| Trial | Dataset | Expected test metrics |
|---|---|---|
| `simsv2_trial159` | CH-SIMS v2 | Acc5 57.25 / Acc3 73.50 / Acc2 80.08 / F1 80.15 / MAE 0.2904 / Corr 0.7107 |
| `mustard_trial1239` | MUStARD | Acc 79.41 / F1 79.39 |
| `mosei_deberta_trial36` | CMU-MOSEI DeBERTa | Acc7 56.24 / Acc2 87.40 / F1 87.37 / MAE 0.4988 / Corr 0.8035 |
| `mosei_bert_trial200` | CMU-MOSEI BERT | Acc7 54.51 / Acc2 85.91 / F1 85.84 / MAE 0.5264 / Corr 0.7710 |
| `mosi_deberta_trial812` | CMU-MOSI DeBERTa | Acc7 49.78 / Acc2 88.85 / F1 88.79 / MAE 0.6004 / Corr 0.8583 |

Argv strings live in [`sota_2026_11_trials.py`](sota_2026_11_trials.py). Training uses `ablation_study/train.py` or `ablation_study/train_classify.py` under conda env `prism`.

```bash
cd My_creation
GPU=1 bash fixed_experiment/run_sota_2026_11.sh mustard_trial1239
python fixed_experiment/train_fixed_sota_2026_11.py --trial simsv2_trial159 --dry-run
python fixed_experiment/train_fixed_sota_2026_11.py --trial simsv2_trial159 --check-only
```

Ablations keep the same frozen argv and change one switch: `ABLATION=no_ib|no_mselector|no_infogate`, `ROUTING=soft|argmax`, or `NO_LRIB=1`. Outputs go to `runs/sota_2026_11/<trial>[_variant]/`. Checkpoints under `runs/` are not committed.
