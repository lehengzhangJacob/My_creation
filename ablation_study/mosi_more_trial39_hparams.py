"""Frozen hyperparameters for CMU-MOSI ``more/mosi`` Optuna trial 39.

This is the MOSI PRISM row used in ``overleaf_69e83a58/acl_latex.tex``:
Acc-7 50.80, Acc-2 89.01, F1 0.8898, MAE 0.6056, Corr 0.8557.
PRISM runs use ``ablation_study/train.py``; non-``none`` modes only append
``--ablation``.

Sources:
  - ``logs/optuna/4090D_restart/more/mosi/db/mosi.db`` —
    study ``infogate_mosi_more_devmae``, trial 39.
  - ``logs/optuna/4090D_restart/more/mosi/train_logs/mosi_more_mosi_trial_39.log``.
"""

from __future__ import annotations


TRIAL_39_PARAMS: dict[str, float | int] = {
    "n_epochs": 99,
    "learning_rate": 2.1771956159852384e-05,
    "ig_learning_rate": 0.00023343084662344783,
    "beta_ib": 21.579529779977843,
    "num_infogate_layers": 4,
    "bottleneck_dim": 192,
    "mse_weight": 1.4287299746044693,
    "dropout_prob": 0.25261088895038614,
    "alpha_ib": 0.005325932479345783,
    "stage1_epochs": 5,
    "warmup_proportion": 0.1015546577959897,
    "weight_decay": 0.0004838783781678784,
    "ema_decay": 0.9952993059827738,
    "selector_target_temp": 0.7462274292205988,
    "selector_rib_weight": 0.07290212324484752,
    "gumbel_tau_start": 1.5648088844745078,
    "gumbel_tau_end": 0.10511429460536832,
    "num_heads": 8,
    "unified_dim": 128,
    "ema_start_epoch": 4,
    "seed": 128,
}

# batch_config 0 in the tier-3 MOSI grid used by more/mosi.
TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEP = 2
IB_HIDDEN_DIM = 128
EARLY_STOP_PATIENCE = 15
SELECTION_METRIC = "mae"


def format_train_float_argv(p: dict[str, float | int]) -> dict[str, str]:
    """Mirror ``optuna_search_v2.objective()`` argv formatting."""
    return {
        "learning_rate": f"{float(p['learning_rate']):.6e}",
        "ig_learning_rate": f"{float(p['ig_learning_rate']):.6e}",
        "beta_ib": f"{float(p['beta_ib']):.4f}",
        "mse_weight": f"{float(p['mse_weight']):.4f}",
        "dropout_prob": f"{float(p['dropout_prob']):.4f}",
        "alpha_ib": f"{float(p['alpha_ib']):.6f}",
        "warmup_proportion": f"{float(p['warmup_proportion']):.4f}",
        "weight_decay": f"{float(p['weight_decay']):.6f}",
        "ema_decay": str(p["ema_decay"]),
        "selector_target_temp": f"{float(p['selector_target_temp']):.4f}",
        "selector_rib_weight": f"{float(p['selector_rib_weight']):.4f}",
        "gumbel_tau_start": f"{float(p['gumbel_tau_start']):.4f}",
        "gumbel_tau_end": f"{float(p['gumbel_tau_end']):.4f}",
    }


def build_train_argv(
    *,
    checkpoint_dir: str,
    ablation: str = "none",
    seed: int | None = None,
    model: str | None = None,
    text_backbone: str | None = None,
    model_variant: str | None = None,
    confidence_mode: str | None = None,
    routing_mode: str | None = None,
    disable_l_rib: bool = False,
) -> list[str]:
    """CLI tokens for ``ablation_study/train.py`` (flag/value pairs)."""
    p = dict(TRIAL_39_PARAMS)
    if seed is not None:
        p["seed"] = int(seed)
    fmt = format_train_float_argv(p)
    argv = [
        "--dataset",
        "mosi",
        "--n_epochs",
        str(int(p["n_epochs"])),
        "--train_batch_size",
        str(TRAIN_BATCH_SIZE),
        "--gradient_accumulation_step",
        str(GRADIENT_ACCUMULATION_STEP),
        "--checkpoint_dir",
        checkpoint_dir,
        "--selection_metric",
        SELECTION_METRIC,
        "--seed",
        str(int(p["seed"])),
        "--learning_rate",
        fmt["learning_rate"],
        "--ig_learning_rate",
        fmt["ig_learning_rate"],
        "--beta_ib",
        fmt["beta_ib"],
        "--num_infogate_layers",
        str(int(p["num_infogate_layers"])),
        "--bottleneck_dim",
        str(int(p["bottleneck_dim"])),
        "--mse_weight",
        fmt["mse_weight"],
        "--dropout_prob",
        fmt["dropout_prob"],
        "--alpha_ib",
        fmt["alpha_ib"],
        "--stage1_epochs",
        str(int(p["stage1_epochs"])),
        "--warmup_proportion",
        fmt["warmup_proportion"],
        "--weight_decay",
        fmt["weight_decay"],
        "--ema_decay",
        fmt["ema_decay"],
        "--selector_target_temp",
        fmt["selector_target_temp"],
        "--selector_rib_weight",
        fmt["selector_rib_weight"],
        "--gumbel_tau_start",
        fmt["gumbel_tau_start"],
        "--gumbel_tau_end",
        fmt["gumbel_tau_end"],
        "--num_heads",
        str(int(p["num_heads"])),
        "--unified_dim",
        str(int(p["unified_dim"])),
        "--ib_hidden_dim",
        str(IB_HIDDEN_DIM),
        "--ema_start_epoch",
        str(int(p["ema_start_epoch"])),
        "--early_stop_patience",
        str(EARLY_STOP_PATIENCE),
    ]
    if ablation != "none":
        argv.extend(["--ablation", ablation])
    for flag, value in (
        ("--model", model),
        ("--text_backbone", text_backbone),
        ("--model_variant", model_variant),
        ("--confidence_mode", confidence_mode),
        ("--routing_mode", routing_mode),
    ):
        if value is not None:
            argv.extend([flag, str(value)])
    if disable_l_rib:
        argv.append("--disable_l_rib")
    return argv
