"""Sensible, round hyperparameters for a DeBERTa-only MOSI/MOSEI baseline.

This recipe is intentionally *not* tied to PRISM Optuna trials. It follows
standard DeBERTa-v3 fine-tuning defaults (lr, weight decay, warmup, dropout)
plus the same training utilities already used by ``ablation_study/train.py``
(L1 + MSE task loss, dev-MAE checkpoint selection, EMA, early stopping).

Only ``--model_variant text_only`` knobs are exercised; InfoGate / DPR / IB
arguments are omitted and fall back to train.py defaults (they are unused).
"""

from __future__ import annotations

# Shared across MOSI and MOSEI for a fair, easy-to-report recipe.
TEXT_ONLY_PARAMS: dict[str, float | int] = {
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "dropout_prob": 0.1,
    "warmup_proportion": 0.1,
    "mse_weight": 1.0,
    "n_epochs": 50,
    "stage1_epochs": 0,
    "ema_decay": 0.999,
    "ema_start_epoch": 1,
    "seed": 128,
}

TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEP = 2  # effective batch 32
EARLY_STOP_PATIENCE = 10
SELECTION_METRIC = "mae"


def build_train_argv(
    *,
    dataset: str,
    checkpoint_dir: str,
    seed: int | None = None,
    model: str | None = None,
) -> list[str]:
    """CLI tokens for ``ablation_study/train.py`` (flag/value pairs)."""
    if dataset not in {"mosi", "mosei"}:
        raise ValueError(f"dataset must be mosi or mosei, got {dataset!r}")

    p = dict(TEXT_ONLY_PARAMS)
    if seed is not None:
        p["seed"] = int(seed)

    return [
        "--dataset",
        dataset,
        "--model_variant",
        "text_only",
        "--text_backbone",
        "deberta",
        "--n_epochs",
        str(int(p["n_epochs"])),
        "--stage1_epochs",
        str(int(p["stage1_epochs"])),
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
        f"{float(p['learning_rate']):.6e}",
        "--weight_decay",
        f"{float(p['weight_decay']):.4f}",
        "--dropout_prob",
        f"{float(p['dropout_prob']):.4f}",
        "--warmup_proportion",
        f"{float(p['warmup_proportion']):.4f}",
        "--mse_weight",
        f"{float(p['mse_weight']):.4f}",
        "--ema_decay",
        str(p["ema_decay"]),
        "--ema_start_epoch",
        str(int(p["ema_start_epoch"])),
        "--early_stop_patience",
        str(EARLY_STOP_PATIENCE),
        *([] if model is None else ["--model", str(model)]),
    ]
