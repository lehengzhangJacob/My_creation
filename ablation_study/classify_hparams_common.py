"""Shared helpers for frozen ``ablation_study/train_classify.py`` runs."""

from __future__ import annotations

MAX_SEQ_LEN = {
    "ur_funny": 64,
    "mustard": 77,
}

CONFIDENCE_MODE_CHOICES = ("token", "modality", "uniform")
ROUTING_MODE_CHOICES = ("gumbel", "argmax", "soft", "batch_gumbel")


def format_classify_float_argv(p: dict[str, float | int]) -> dict[str, str]:
    """Mirror ``optuna_search_classify`` train subprocess argv formatting."""
    return {
        "learning_rate": f"{float(p['learning_rate']):.6e}",
        "ig_learning_rate": f"{float(p['ig_learning_rate']):.6e}",
        "beta_ib": f"{float(p['beta_ib']):.4f}",
        "alpha_ib": f"{float(p['alpha_ib']):.6f}",
        "dropout_prob": f"{float(p['dropout_prob']):.4f}",
        "warmup_proportion": f"{float(p['warmup_proportion']):.4f}",
        "weight_decay": f"{float(p['weight_decay']):.6f}",
        "ema_decay": str(p["ema_decay"]),
        "gumbel_tau_start": f"{float(p['gumbel_tau_start']):.4f}",
        "gumbel_tau_end": f"{float(p['gumbel_tau_end']):.4f}",
        "selector_target_temp": f"{float(p['selector_target_temp']):.4f}",
        "selector_rib_weight": f"{float(p['selector_rib_weight']):.4f}",
        "ib_loss_mult_end": f"{float(p['ib_loss_mult_end']):.6f}",
        "focal_gamma": f"{float(p['focal_gamma']):.6f}",
        "rdrop_alpha": f"{float(p['rdrop_alpha']):.6f}",
        "selector_balance_weight": f"{float(p['selector_balance_weight']):.6f}",
    }


def build_classify_argv(
    *,
    dataset: str,
    params: dict[str, float | int | bool | str],
    checkpoint_dir: str,
    selection_metric: str = "binary_acc",
    ablation: str = "none",
    seed: int | None = None,
    disable_hcf: bool = False,
    confidence_mode: str | None = None,
    routing_mode: str | None = None,
) -> list[str]:
    """CLI tokens for ``ablation_study/train_classify.py`` (cwd = ``My_creation``)."""
    if dataset not in MAX_SEQ_LEN:
        raise ValueError(f"unsupported dataset {dataset!r}")
    if confidence_mode is not None and confidence_mode not in CONFIDENCE_MODE_CHOICES:
        raise ValueError(
            f"unsupported confidence_mode {confidence_mode!r}; "
            f"expected one of {CONFIDENCE_MODE_CHOICES}")
    if routing_mode is not None and routing_mode not in ROUTING_MODE_CHOICES:
        raise ValueError(
            f"unsupported routing_mode {routing_mode!r}; "
            f"expected one of {ROUTING_MODE_CHOICES}")

    p = dict(params)
    if seed is not None:
        p["seed"] = int(seed)
    fmt = format_classify_float_argv(p)
    ib_hidden_dim = int(p.get("ib_hidden_dim", p["unified_dim"]))

    argv = [
        "--dataset",
        dataset,
        "--max_seq_length",
        str(MAX_SEQ_LEN[dataset]),
        "--n_epochs",
        str(int(p["n_epochs"])),
        "--stage1_epochs",
        str(int(p["stage1_epochs"])),
        "--train_batch_size",
        str(int(p["train_batch_size"])),
        "--gradient_accumulation_step",
        str(int(p["gradient_accumulation_step"])),
        "--learning_rate",
        fmt["learning_rate"],
        "--ig_learning_rate",
        fmt["ig_learning_rate"],
        "--beta_ib",
        fmt["beta_ib"],
        "--alpha_ib",
        fmt["alpha_ib"],
        "--bottleneck_dim",
        str(int(p["bottleneck_dim"])),
        "--num_infogate_layers",
        str(int(p["num_infogate_layers"])),
        "--unified_dim",
        str(int(p["unified_dim"])),
        "--ib_hidden_dim",
        str(ib_hidden_dim),
        "--num_heads",
        str(int(p["num_heads"])),
        "--dropout_prob",
        fmt["dropout_prob"],
        "--warmup_proportion",
        fmt["warmup_proportion"],
        "--weight_decay",
        fmt["weight_decay"],
        "--ema_decay",
        fmt["ema_decay"],
        "--ema_start_epoch",
        str(int(p["ema_start_epoch"])),
        "--gumbel_tau_start",
        fmt["gumbel_tau_start"],
        "--gumbel_tau_end",
        fmt["gumbel_tau_end"],
        "--selector_target_temp",
        fmt["selector_target_temp"],
        "--selector_rib_weight",
        fmt["selector_rib_weight"],
        "--selection_metric",
        selection_metric,
        "--checkpoint_dir",
        checkpoint_dir,
        "--seed",
        str(int(p["seed"])),
        "--ib_loss_mult_end",
        fmt["ib_loss_mult_end"],
        "--bce_pos_weight_mode",
        str(p["bce_pos_weight_mode"]),
        "--focal_gamma",
        fmt["focal_gamma"],
        "--rdrop_alpha",
        fmt["rdrop_alpha"],
        "--early_stop_patience",
        str(int(p["early_stop_patience"])),
        "--selection_smooth_window",
        str(int(p["selection_smooth_window"])),
        "--selector_balance_weight",
        fmt["selector_balance_weight"],
    ]
    if p.get("freeze_backbone_stage2"):
        argv.append("--freeze_backbone_stage2")
    if ablation != "none":
        argv.extend(["--ablation", ablation])
    if disable_hcf:
        argv.append("--disable_hcf")
    if confidence_mode is not None:
        argv.extend(["--confidence_mode", confidence_mode])
    if routing_mode is not None:
        argv.extend(["--routing_mode", routing_mode])
    return argv
