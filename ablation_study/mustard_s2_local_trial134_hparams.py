"""Frozen hyperparameters for MUStARD Optuna s2_local trial 134 (paper classification row).

Sources (keep in sync when regenerating):
  - Study: ``infogate_mustard_optuna_classify_albert_hcf_20260426_192758_s2_local``
  - DB: ``.../db/mustard_infogate_mustard_optuna_classify_albert_hcf_20260426_192758_s2_local.db``
  - Train log: ``.../train_logs/mustard_s2_local_trial_134.log``
  - Best Results (dev ``binary_acc`` selection): test Acc **79.41%**, F1 **79.34%**
  - Paper: ``acl_latex.tex`` / ``mhd_msd.py`` — **79.4%** (rounded from 79.41%)
"""

from __future__ import annotations

from classify_hparams_common import build_classify_argv

TRIAL_134_PARAMS: dict[str, float | int | bool | str] = {
    "n_epochs": 60,
    "stage1_epochs": 12,
    "train_batch_size": 16,
    "gradient_accumulation_step": 1,
    "learning_rate": 2.5540061141509894e-05,
    "ig_learning_rate": 9.313311202294946e-05,
    "beta_ib": 4.718572378492174,
    "num_infogate_layers": 4,
    "bottleneck_dim": 192,
    "dropout_prob": 0.22006483381853303,
    "alpha_ib": 0.01251565337799992,
    "warmup_proportion": 0.19292820569036706,
    "weight_decay": 0.0003811390821608703,
    "ema_decay": 0.99,
    "ema_start_epoch": 5,
    "selector_target_temp": 0.6580348766780887,
    "selector_rib_weight": 0.10372224080405697,
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.5,
    "num_heads": 4,
    "unified_dim": 256,
    "ib_loss_mult_end": 1.0,
    "freeze_backbone_stage2": False,
    "focal_gamma": 0.0,
    "rdrop_alpha": 0.0,
    "bce_pos_weight_mode": "none",
    "early_stop_patience": 0,
    "selection_smooth_window": 1,
    "selector_balance_weight": 0.0,
    "seed": 42,
}

SELECTION_METRIC = "binary_acc"

PAPER_TEST_METRICS = {
    "acc_pct": 79.41,
    "f1_pct": 79.34,
    "dev_selection_score": 0.735294,
}


def build_train_argv(
    *,
    checkpoint_dir: str,
    ablation: str = "none",
    seed: int | None = None,
    disable_hcf: bool = False,
    confidence_mode: str | None = None,
    routing_mode: str | None = None,
) -> list[str]:
    return build_classify_argv(
        dataset="mustard",
        params=TRIAL_134_PARAMS,
        checkpoint_dir=checkpoint_dir,
        selection_metric=SELECTION_METRIC,
        ablation=ablation,
        seed=seed,
        disable_hcf=disable_hcf,
        confidence_mode=confidence_mode,
        routing_mode=routing_mode,
    )
