"""
InfoGate + ALBERT integration (classification ablations).

Weights load from ``My_creation/albert-base-v2``. ``ablation`` is passed into
``ablation_study/infogate_modules.InfoGate`` (``none`` = full PRISM).
"""

from __future__ import annotations

import os

from transformers.models.albert.modeling_albert import (
    AlbertPreTrainedModel, AlbertModel,
)
from infogate_modules import InfoGate
import global_configs
from global_configs import DEVICE

_ABL_DIR = os.path.dirname(os.path.abspath(__file__))
_MY_CREATION_DIR = os.path.join(_ABL_DIR, "..")
_MODEL_DIR = os.path.join(_MY_CREATION_DIR, "albert-base-v2")


def _resolve_dims(config, mc):
    text_dim = getattr(mc, 'text_dim', None) or global_configs.TEXT_DIM
    acoustic_dim = getattr(mc, 'acoustic_dim', None) or global_configs.ACOUSTIC_DIM
    visual_dim = getattr(mc, 'visual_dim', None) or global_configs.VISUAL_DIM
    configured_hcf_dim = getattr(mc, 'hcf_dim', None)
    hcf_dim = (global_configs.HCF_DIM
               if configured_hcf_dim is None else int(configured_hcf_dim))

    ds = getattr(mc, 'dataset', None)
    if ds and (text_dim <= 0 or acoustic_dim <= 0 or visual_dim <= 0):
        global_configs.set_dataset_config(ds)
        text_dim = getattr(mc, 'text_dim', None) or global_configs.TEXT_DIM
        acoustic_dim = getattr(mc, 'acoustic_dim', None) or global_configs.ACOUSTIC_DIM
        visual_dim = getattr(mc, 'visual_dim', None) or global_configs.VISUAL_DIM
        hcf_dim = (global_configs.HCF_DIM
                   if configured_hcf_dim is None else int(configured_hcf_dim))

    if text_dim <= 0:
        text_dim = config.hidden_size
    if acoustic_dim <= 0 or visual_dim <= 0:
        raise ValueError(
            "Acoustic / visual dims not configured. "
            "Call global_configs.set_dataset_config(...) before model creation."
        )
    return text_dim, acoustic_dim, visual_dim, hcf_dim


class InfoGate_AlbertModel(AlbertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM, HCF_DIM = _resolve_dims(config, multimodal_config)
        self.config = config

        model = AlbertModel.from_pretrained(_MODEL_DIR)
        self.model = model.to(DEVICE)

        ig_args = {
            'text_dim': TEXT_DIM,
            'acoustic_dim': ACOUSTIC_DIM,
            'visual_dim': VISUAL_DIM,
            'hcf_dim': HCF_DIM,
            'unified_dim': getattr(multimodal_config, 'unified_dim', 256),
            'ib_hidden_dim': getattr(multimodal_config, 'ib_hidden_dim', 256),
            'bottleneck_dim': getattr(multimodal_config, 'bottleneck_dim', 128),
            'num_heads': getattr(multimodal_config, 'num_heads', 4),
            'num_infogate_layers': getattr(multimodal_config, 'num_infogate_layers', 3),
            'dropout_prob': getattr(multimodal_config, 'dropout_prob', 0.1),
            'beta_ib': getattr(multimodal_config, 'beta_ib', 32),
            'alpha_ib': getattr(multimodal_config, 'alpha_ib', 0.01),
            'use_l_lib': getattr(multimodal_config, 'use_l_lib', True),
            'use_l_tran': getattr(multimodal_config, 'use_l_tran', True),
            'use_l_rib': getattr(multimodal_config, 'use_l_rib', True),
            'selector_target_temp': getattr(multimodal_config, 'selector_target_temp', 0.35),
            'selector_balance_weight': getattr(multimodal_config, 'selector_balance_weight', 0.0),
            'selector_rib_weight': getattr(multimodal_config, 'selector_rib_weight', 0.05),
            'gumbel_tau': getattr(multimodal_config, 'gumbel_tau_start', 1.0),
            'task_type': getattr(multimodal_config, 'task_type', 'binary'),
            'ablation': getattr(multimodal_config, 'ablation', 'none'),
            'confidence_mode': getattr(multimodal_config, 'confidence_mode', 'token'),
            'routing_mode': getattr(multimodal_config, 'routing_mode', 'gumbel'),
        }

        self.infogate = InfoGate(ig_args)
        self.init_weights()

    def forward(self, input_ids, visual, acoustic, hcf=None,
                labels=None, stage=1):
        pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        attention_mask = input_ids.ne(pad_id).long()

        text_features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]  # [B, T, 768]

        logits, ib_loss, loss_dict, h_p = self.infogate(
            text_features, acoustic, visual,
            hcf=hcf,
            labels=labels, stage=stage,
            attention_mask=attention_mask,
        )
        return logits, ib_loss, loss_dict, h_p


class InfoGate_AlbertForSequenceClassification(AlbertPreTrainedModel):
    """Wrapper so ``train_classify.py`` can distinguish ALBERT vs DeBERTa param groups."""

    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.albert = InfoGate_AlbertModel(config, multimodal_config)

    def forward(self, input_ids, visual, acoustic, hcf=None,
                labels=None, stage=1):
        return self.albert(
            input_ids, visual, acoustic, hcf=hcf,
            labels=labels, stage=stage,
        )
