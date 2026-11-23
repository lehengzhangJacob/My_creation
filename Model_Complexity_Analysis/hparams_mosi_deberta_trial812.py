"""Paper MOSI DeBERTa — sota_search trial 812."""
from __future__ import annotations


def build_train_argv(checkpoint_dir: str) -> list[str]:
    return [
        "--dataset", "mosi",
        "--n_epochs", "99",
        "--train_batch_size", "16",
        "--gradient_accumulation_step", "2",
        "--checkpoint_dir", checkpoint_dir,
        "--selection_metric", "mae",
        "--seed", "3812",
        "--learning_rate", "1.529548e-05",
        "--ig_learning_rate", "2.547068e-04",
        "--beta_ib", "20.8832",
        "--num_infogate_layers", "5",
        "--bottleneck_dim", "192",
        "--mse_weight", "1.5603",
        "--dropout_prob", "0.2156",
        "--alpha_ib", "0.008455",
        "--stage1_epochs", "2",
        "--warmup_proportion", "0.0921",
        "--weight_decay", "0.000263",
        "--ema_decay", "0.9952993059827738",
        "--selector_target_temp", "0.7454",
        "--selector_rib_weight", "0.0889",
        "--gumbel_tau_start", "1.3790",
        "--gumbel_tau_end", "0.1053",
        "--num_heads", "8",
        "--unified_dim", "128",
        "--ib_hidden_dim", "128",
        "--ema_start_epoch", "4",
        "--early_stop_patience", "15",
    ]
