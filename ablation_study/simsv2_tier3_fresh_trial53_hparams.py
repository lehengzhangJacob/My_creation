"""Frozen hyperparameters for CH-SIMS v2 Optuna tier3_fresh trial 53.

PRISM ablation runs use ``ablation_study/train.py`` so code and outputs stay
under this frozen ablation bundle. Numeric knobs mirror
``fixed_experiment/simsv2_tier3_fresh_trial53_hparams.py``.
"""

from __future__ import annotations

# Params from Optuna trial 53 (sqlite trial_params + log header).
TRIAL_53_PARAMS: dict[str, float | int] = {
    "n_epochs": 61,
    "learning_rate": 2.4304567421655e-05,
    "ig_learning_rate": 0.000907583956926654,
    "beta_ib": 12.3622402899691,
    "num_infogate_layers": 3,
    "bottleneck_dim": 64,
    "mse_weight": 0.572877136929496,
    "dropout_prob": 0.194106519245586,
    "alpha_ib": 0.00129998678433247,
    "stage1_epochs": 3,
    "warmup_proportion": 0.142570422565119,
    "weight_decay": 0.00175555929968471,
    "ema_decay": 0.997372716935179,
    "selector_target_temp": 0.792342876228235,
    "selector_rib_weight": 0.106334681848934,
    "gumbel_tau_start": 1.72024501861536,
    "gumbel_tau_end": 0.248170386122028,
    "num_heads": 8,
    "unified_dim": 384,
    "ema_start_epoch": 10,
    "seed": 128,
}

TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEP = 4
IB_HIDDEN_DIM = int(TRIAL_53_PARAMS["unified_dim"])
EARLY_STOP_PATIENCE = 15
SELECTION_METRIC = "mae"

PAPER_TEST_METRICS = {
    "acc5": 55.6,
    "acc3": 73.4,
    "acc2": 80.1,
    "f1": 80.1,
    "mae": 0.291,
    "corr": 0.705,
}


def format_train_float_argv(p: dict[str, float | int]) -> dict[str, str]:
    """Mirror ``optuna_search_v2.objective()`` argv formatting for floats."""
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
    p = dict(TRIAL_53_PARAMS)
    if seed is not None:
        p["seed"] = int(seed)
    fmt = format_train_float_argv(p)
    argv = [
        "--dataset",
        "simsv2",
        "--simsv2_feature_mode",
        "mmsa",
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
