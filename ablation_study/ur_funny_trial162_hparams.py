"""Frozen hyperparameters for UR-FUNNY v2 Optuna trial 162 (paper classification row).

Sources (keep in sync when regenerating):
  - Study: ``infogate_ur_funny_optuna_classify_albert_hcf_20260426_192758``
  - DB: ``logs/optuna/4090D_restart/classification/optuna_classify_albert_hcf_20260426_192758/db/ur_funny.db``
  - Train log: ``.../train_logs/ur_funny_trial_162.log``
  - Best Results (dev ``binary_acc`` selection): test Acc **75.15%**, F1 **75.01%**
  - Paper: ``acl_latex.tex`` / ``mhd_msd.py`` — **75.2%** (rounded from 75.15%)
"""

from __future__ import annotations

from classify_hparams_common import build_classify_argv

TRIAL_162_PARAMS: dict[str, float | int | bool | str] = {
    "n_epochs": 41,
    "stage1_epochs": 9,
    "train_batch_size": 16,
    "gradient_accumulation_step": 4,
    "learning_rate": 5.252010846850622e-06,
    "ig_learning_rate": 0.002279562206475482,
    "beta_ib": 20.469299809706353,
    "num_infogate_layers": 5,
    "bottleneck_dim": 96,
    "dropout_prob": 0.23430823340719972,
    "alpha_ib": 0.0014638127620707226,
    "warmup_proportion": 0.19830100461004277,
    "weight_decay": 0.016202597927225556,
    "ema_decay": 0.999,
    "ema_start_epoch": 5,
    "selector_target_temp": 0.5828024468586878,
    "selector_rib_weight": 0.049789656342925113,
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
    "acc_pct": 75.15,
    "f1_pct": 75.01,
    "dev_selection_score": 0.744898,
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
        dataset="ur_funny",
        params=TRIAL_162_PARAMS,
        checkpoint_dir=checkpoint_dir,
        selection_metric=SELECTION_METRIC,
        ablation=ablation,
        seed=seed,
        disable_hcf=disable_hcf,
        confidence_mode=confidence_mode,
        routing_mode=routing_mode,
    )
