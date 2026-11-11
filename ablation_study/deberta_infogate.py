"""
InfoGate + DeBERTa integration module.
Uses DeBERTa-v3-base as the text encoder; prediction is handled by InfoGate's
internal MLP head. Weights load from ``My_creation/deberta-v3-base`` (same as
``My_creation/deberta_infogate.py``); use ``DebertaV2Model.from_pretrained`` +
``.to(DEVICE)`` for transformers 4.29.x (ITHP5090 stack).

``ablation == 'none'`` uses ``infogate_modules_fixed.InfoGate`` for frozen Optuna
parity; other ``--ablation`` modes use ``infogate_modules.InfoGate``.
"""

from __future__ import annotations

import os

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
)
from infogate_modules import InfoGate as InfoGatePrism
from infogate_modules_fixed import InfoGate as InfoGateFixed
import global_configs
from global_configs import DEVICE

_ABL_DIR = os.path.dirname(os.path.abspath(__file__))
_MY_CREATION_DIR = os.path.dirname(_ABL_DIR)
_MODEL_DIR = os.path.join(_MY_CREATION_DIR, "deberta-v3-base")


def _resolve_dims(config, mc):
    text_dim = getattr(mc, "text_dim", None) or global_configs.TEXT_DIM
    acoustic_dim = getattr(mc, "acoustic_dim", None) or global_configs.ACOUSTIC_DIM
    visual_dim = getattr(mc, "visual_dim", None) or global_configs.VISUAL_DIM

    ds = getattr(mc, "dataset", None)
    if ds and (text_dim <= 0 or acoustic_dim <= 0 or visual_dim <= 0):
        global_configs.set_dataset_config(ds)
        text_dim = getattr(mc, "text_dim", None) or global_configs.TEXT_DIM
        acoustic_dim = getattr(mc, "acoustic_dim", None) or global_configs.ACOUSTIC_DIM
        visual_dim = getattr(mc, "visual_dim", None) or global_configs.VISUAL_DIM

    if text_dim <= 0:
        text_dim = config.hidden_size
    if acoustic_dim <= 0 or visual_dim <= 0:
        raise ValueError(
            "Acoustic / visual dims not configured. "
            "Call global_configs.set_dataset_config(...) before model creation."
        )
    return text_dim, acoustic_dim, visual_dim


class InfoGate_DebertaModel(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = _resolve_dims(config, multimodal_config)
        self.config = config

        model = DebertaV2Model.from_pretrained(_MODEL_DIR)
        self.model = model.to(DEVICE)

        ig_args = {
            "text_dim": TEXT_DIM,
            "acoustic_dim": ACOUSTIC_DIM,
            "visual_dim": VISUAL_DIM,
            "unified_dim": getattr(multimodal_config, "unified_dim", 256),
            "ib_hidden_dim": getattr(multimodal_config, "ib_hidden_dim", 256),
            "bottleneck_dim": getattr(multimodal_config, "bottleneck_dim", 128),
            "num_heads": getattr(multimodal_config, "num_heads", 4),
            "num_infogate_layers": getattr(multimodal_config, "num_infogate_layers", 3),
            "dropout_prob": getattr(multimodal_config, "dropout_prob", 0.1),
            "beta_ib": getattr(multimodal_config, "beta_ib", 32),
            "alpha_ib": getattr(multimodal_config, "alpha_ib", 0.01),
            "use_l_lib": getattr(multimodal_config, "use_l_lib", True),
            "use_l_tran": getattr(multimodal_config, "use_l_tran", True),
            "use_l_rib": getattr(multimodal_config, "use_l_rib", True),
            "selector_target_temp": getattr(multimodal_config, "selector_target_temp", 0.35),
            "selector_balance_weight": getattr(multimodal_config, "selector_balance_weight", 0.0),
            "selector_rib_weight": getattr(multimodal_config, "selector_rib_weight", 0.05),
            "gumbel_tau": getattr(multimodal_config, "gumbel_tau_start", 1.0),
            "confidence_mode": getattr(multimodal_config, "confidence_mode", "token"),
            "routing_mode": getattr(multimodal_config, "routing_mode", "gumbel"),
            "task_type": getattr(multimodal_config, "task_type", "regression"),
            "ablation": getattr(multimodal_config, "ablation", "none"),
        }

        _abl = ig_args.get("ablation", "none")
        if _abl == "none":
            ig_fixed = {k: v for k, v in ig_args.items() if k != "ablation"}
            self.infogate = InfoGateFixed(ig_fixed)
        else:
            self.infogate = InfoGatePrism(ig_args)
        self.init_weights()

    def forward(self, input_ids, visual, acoustic, labels=None, stage=1):
        pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        attention_mask = input_ids.ne(pad_id).long()

        text_features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]  # [B, T, 768]

        logits, ib_loss, loss_dict, h_p = self.infogate(
            text_features,
            acoustic,
            visual,
            labels=labels,
            stage=stage,
            attention_mask=attention_mask,
        )
        return logits, ib_loss, loss_dict, h_p


class InfoGate_DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.dberta = InfoGate_DebertaModel(config, multimodal_config)

    def forward(self, input_ids, visual, acoustic, labels=None, stage=1):
        return self.dberta(
            input_ids,
            visual,
            acoustic,
            labels=labels,
            stage=stage,
        )
