"""Paper CH-SIMS v2 — sota_search trial 159."""
from __future__ import annotations


def build_train_argv(checkpoint_dir: str) -> list[str]:
    return [
        "--dataset", "simsv2",
        "--simsv2_feature_mode", "mmsa",
        "--n_epochs", "61",
        "--train_batch_size", "8",
        "--gradient_accumulation_step", "4",
        "--checkpoint_dir", checkpoint_dir,
        "--selection_metric", "mae",
        "--seed", "976",
        "--learning_rate", "1.534527e-05",
        "--ig_learning_rate", "1.407749e-03",
        "--beta_ib", "14.8506",
        "--num_infogate_layers", "4",
        "--bottleneck_dim", "64",
        "--mse_weight", "0.6618",
        "--dropout_prob", "0.1546",
        "--alpha_ib", "0.001295",
        "--stage1_epochs", "4",
        "--warmup_proportion", "0.1223",
        "--weight_decay", "0.001838",
        "--ema_decay", "0.997372716935179",
        "--selector_target_temp", "0.8698",
        "--selector_rib_weight", "0.0991",
        "--gumbel_tau_start", "1.3813",
        "--gumbel_tau_end", "0.2378",
        "--num_heads", "8",
        "--unified_dim", "384",
        "--ib_hidden_dim", "384",
        "--ema_start_epoch", "10",
        "--early_stop_patience", "15",
    ]
