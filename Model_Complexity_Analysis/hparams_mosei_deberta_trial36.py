"""Paper MOSEI DeBERTa — sota_search trial 36."""
from __future__ import annotations


def build_train_argv(checkpoint_dir: str) -> list[str]:
    return [
        "--dataset", "mosei",
        "--n_epochs", "50",
        "--train_batch_size", "4",
        "--gradient_accumulation_step", "8",
        "--checkpoint_dir", checkpoint_dir,
        "--selection_metric", "mae",
        "--seed", "4861",
        "--learning_rate", "2.047660e-05",
        "--ig_learning_rate", "4.221805e-04",
        "--beta_ib", "19.2729",
        "--num_infogate_layers", "2",
        "--bottleneck_dim", "64",
        "--mse_weight", "2.2427",
        "--dropout_prob", "0.2530",
        "--alpha_ib", "0.009207",
        "--stage1_epochs", "11",
        "--warmup_proportion", "0.1156",
        "--weight_decay", "0.000948",
        "--ema_decay", "0.999",
        "--selector_target_temp", "0.4076",
        "--selector_rib_weight", "0.0420",
        "--gumbel_tau_start", "0.9798",
        "--gumbel_tau_end", "0.4381",
        "--num_heads", "4",
        "--unified_dim", "256",
        "--ib_hidden_dim", "256",
        "--ema_start_epoch", "5",
        "--early_stop_patience", "10",
    ]
