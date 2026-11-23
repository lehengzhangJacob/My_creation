"""Frozen trials from the Nov 2026 re-search on the 4x RTX 3090 host (``sota_search/``).

Each entry replays the exact command of an Optuna trial that met the paper targets
(every metric >= the paper value and strictly above the best baseline). The argv
strings are copied verbatim from ``sota_search/runs/<row>/trial_<n>/cmd.json`` so
float formatting is identical; only ``--checkpoint_dir`` is substituted at launch.

Scripts are the ``ablation_study`` trainers the search used (cwd = ``My_creation``).
Environment of the original runs: conda env ``prism`` (Python 3.10, torch 2.9.1,
transformers 4.29.2), GPU RTX 3090.
"""

from __future__ import annotations

TRIALS: dict[str, dict] = {
    "simsv2_trial159": {
        "row": "CH-SIMS v2",
        "source": "sota_search/runs/simsv2/trial_159",
        "script": "ablation_study/train.py",
        "argv": (
            "--dataset simsv2 --simsv2_feature_mode mmsa --n_epochs 61 --train_batch_size 8 "
            "--gradient_accumulation_step 4 --checkpoint_dir {ckpt} --selection_metric mae "
            "--seed 976 --learning_rate 1.534527e-05 --ig_learning_rate 1.407749e-03 "
            "--beta_ib 14.8506 --num_infogate_layers 4 --bottleneck_dim 64 --mse_weight 0.6618 "
            "--dropout_prob 0.1546 --alpha_ib 0.001295 --stage1_epochs 4 "
            "--warmup_proportion 0.1223 --weight_decay 0.001838 --ema_decay 0.997372716935179 "
            "--selector_target_temp 0.8698 --selector_rib_weight 0.0991 --gumbel_tau_start 1.3813 "
            "--gumbel_tau_end 0.2378 --num_heads 8 --unified_dim 384 --ib_hidden_dim 384 "
            "--ema_start_epoch 10 --early_stop_patience 15"
        ),
        "test": {"Acc5": 57.25, "Acc3": 73.50, "Acc2": 80.08, "F1": 80.15,
                 "MAE": 0.2904, "Corr": 0.7107},
        "runtime": "~54 min",
    },
    "ur_funny_trial197": {
        "row": "UR-FUNNY v2",
        "source": "sota_search/runs/ur_funny/trial_197",
        "script": "ablation_study/train_classify.py",
        "argv": (
            "--dataset ur_funny --max_seq_length 64 --n_epochs 41 --stage1_epochs 9 "
            "--train_batch_size 16 --gradient_accumulation_step 4 --learning_rate 5.553549e-06 "
            "--ig_learning_rate 1.916258e-03 --beta_ib 17.2958 --alpha_ib 0.000961 "
            "--bottleneck_dim 128 --num_infogate_layers 5 --unified_dim 384 --ib_hidden_dim 384 "
            "--num_heads 4 --dropout_prob 0.2191 --warmup_proportion 0.1973 "
            "--weight_decay 0.015704 --ema_decay 0.999 --ema_start_epoch 5 "
            "--gumbel_tau_start 0.9519 --gumbel_tau_end 0.6117 --selector_target_temp 0.6290 "
            "--selector_rib_weight 0.0526 --selection_metric binary_acc --checkpoint_dir {ckpt} "
            "--seed 1312 --ib_loss_mult_end 1.000000 --bce_pos_weight_mode none "
            "--focal_gamma 0.000000 --rdrop_alpha 0.000000 --early_stop_patience 0 "
            "--selection_smooth_window 1 --selector_balance_weight 0.000000"
        ),
        "test": {"Acc": 75.15, "F1": 74.98},
        "runtime": "~2.6 h",
    },
    "mustard_trial1239": {
        "row": "MUStARD",
        "source": "sota_search/runs/mustard/trial_1239",
        "script": "ablation_study/train_classify.py",
        "argv": (
            "--dataset mustard --max_seq_length 77 --n_epochs 60 --stage1_epochs 9 "
            "--train_batch_size 16 --gradient_accumulation_step 1 --learning_rate 1.977148e-05 "
            "--ig_learning_rate 1.571708e-04 --beta_ib 5.1787 --alpha_ib 0.008988 "
            "--bottleneck_dim 128 --num_infogate_layers 4 --unified_dim 256 --ib_hidden_dim 256 "
            "--num_heads 4 --dropout_prob 0.2103 --warmup_proportion 0.2288 "
            "--weight_decay 0.000493 --ema_decay 0.99 --ema_start_epoch 5 "
            "--gumbel_tau_start 1.1299 --gumbel_tau_end 0.4429 --selector_target_temp 0.5683 "
            "--selector_rib_weight 0.0850 --selection_metric binary_acc --checkpoint_dir {ckpt} "
            "--seed 9871 --ib_loss_mult_end 1.000000 --bce_pos_weight_mode none "
            "--focal_gamma 0.000000 --rdrop_alpha 0.000000 --early_stop_patience 0 "
            "--selection_smooth_window 1 --selector_balance_weight 0.000000"
        ),
        "test": {"Acc": 79.41, "F1": 79.39},
        "runtime": "~11 min",
    },
    # Paper MOSEI BERT: Nov search trial 200 (seed 9803).
    "mosei_bert_trial200": {
        "row": "CMU-MOSEI (BERT, paper)",
        "source": "sota_search/runs/mosei_bert/trial_200",
        "script": "ablation_study/train.py",
        "argv": (
            "--dataset mosei --n_epochs 50 --train_batch_size 4 --gradient_accumulation_step 8 "
            "--checkpoint_dir {ckpt} --selection_metric mae --seed 9803 "
            "--learning_rate 2.110924e-05 --ig_learning_rate 2.922366e-04 --beta_ib 22.0786 "
            "--num_infogate_layers 2 --bottleneck_dim 64 --mse_weight 2.3402 --dropout_prob 0.2692 "
            "--alpha_ib 0.005222 --stage1_epochs 11 --warmup_proportion 0.0919 "
            "--weight_decay 0.001521 --ema_decay 0.999 --selector_target_temp 0.4101 "
            "--selector_rib_weight 0.0605 --gumbel_tau_start 1.2039 --gumbel_tau_end 0.4954 "
            "--num_heads 8 --unified_dim 384 --ib_hidden_dim 384 --ema_start_epoch 5 "
            "--model {model} --text_backbone bert --early_stop_patience 10"
        ),
        "test": {"Acc7": 54.51, "Acc2": 85.91, "F1": 85.84, "MAE": 0.5264, "Corr": 0.7710},
        "runtime": "~8.5 h",
        "python_env": "prism",
    },
    "mosei_deberta_trial36": {
        "row": "CMU-MOSEI (DeBERTa)",
        "source": "sota_search/runs/mosei_deberta/trial_36",
        "script": "ablation_study/train.py",
        "argv": (
            "--dataset mosei --n_epochs 50 --train_batch_size 4 --gradient_accumulation_step 8 "
            "--checkpoint_dir {ckpt} --selection_metric mae --seed 4861 "
            "--learning_rate 2.047660e-05 --ig_learning_rate 4.221805e-04 --beta_ib 19.2729 "
            "--num_infogate_layers 2 --bottleneck_dim 64 --mse_weight 2.2427 "
            "--dropout_prob 0.2530 --alpha_ib 0.009207 --stage1_epochs 11 "
            "--warmup_proportion 0.1156 --weight_decay 0.000948 --ema_decay 0.999 "
            "--selector_target_temp 0.4076 --selector_rib_weight 0.0420 "
            "--gumbel_tau_start 0.9798 --gumbel_tau_end 0.4381 --num_heads 4 --unified_dim 256 "
            "--ib_hidden_dim 256 --ema_start_epoch 5 --early_stop_patience 10"
        ),
        "test": {"Acc7": 56.24, "Acc2": 87.40, "F1": 87.37, "MAE": 0.4988, "Corr": 0.8035},
        "runtime": "~11.7 h",
    },
    "mosi_deberta_trial812": {
        "row": "CMU-MOSI (DeBERTa)",
        "source": "sota_search/runs/mosi_deberta/trial_812",
        "script": "ablation_study/train.py",
        "argv": (
            "--dataset mosi --n_epochs 99 --train_batch_size 16 --gradient_accumulation_step 2 "
            "--checkpoint_dir {ckpt} --selection_metric mae --seed 3812 "
            "--learning_rate 1.529548e-05 --ig_learning_rate 2.547068e-04 --beta_ib 20.8832 "
            "--num_infogate_layers 5 --bottleneck_dim 192 --mse_weight 1.5603 "
            "--dropout_prob 0.2156 --alpha_ib 0.008455 --stage1_epochs 2 "
            "--warmup_proportion 0.0921 --weight_decay 0.000263 --ema_decay 0.9952993059827738 "
            "--selector_target_temp 0.7454 --selector_rib_weight 0.0889 "
            "--gumbel_tau_start 1.3790 --gumbel_tau_end 0.1053 --num_heads 8 --unified_dim 128 "
            "--ib_hidden_dim 128 --ema_start_epoch 4 --early_stop_patience 15"
        ),
        "test": {"Acc7": 49.78, "Acc2": 88.85, "F1": 88.79, "MAE": 0.6004, "Corr": 0.8583},
        "runtime": "search run",
    },
}


def build_argv(name: str, checkpoint_dir: str, model_dir: str | None = None) -> list[str]:
    """Expand ``{ckpt}`` / ``{model}`` placeholders. ``model_dir`` defaults to
    ``My_creation/bert-base-uncase`` (paper BERT rows)."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    model = model_dir or str(root / "bert-base-uncase")
    return TRIALS[name]["argv"].format(ckpt=checkpoint_dir, model=model).split()
