"""
InfoGate + BERT integration module.
Uses BERT as the text encoder; prediction is handled by InfoGate's internal MLP head.

Relative ``multimodal_config.model`` names resolve under ``My_creation/`` (same layout
as ``ablation_study/train.py`` path patches).
"""

from __future__ import annotations

import os

from transformers import BertPreTrainedModel, BertModel
from infogate_modules import InfoGate as InfoGatePrism
from infogate_modules_fixed import InfoGate as InfoGateFixed
import global_configs
from global_configs import DEVICE

_ABL_DIR = os.path.dirname(os.path.abspath(__file__))
_MY_CREATION_DIR = os.path.dirname(_ABL_DIR)


def _resolve_bert_pretrained_dir(mc) -> str:
    raw = getattr(mc, "model", "bert-base-chinese")
    if os.path.isabs(raw):
        return raw
    base = os.path.basename(raw.rstrip(os.sep))
    return os.path.join(_MY_CREATION_DIR, base)

def _resolve_dims(config, mc):
    text_dim = getattr(mc, 'text_dim', None) or global_configs.TEXT_DIM
    acoustic_dim = getattr(mc, 'acoustic_dim', None) or global_configs.ACOUSTIC_DIM
    visual_dim = getattr(mc, 'visual_dim', None) or global_configs.VISUAL_DIM

    ds = getattr(mc, 'dataset', None)
    if ds and (text_dim <= 0 or acoustic_dim <= 0 or visual_dim <= 0):
        global_configs.set_dataset_config(ds)
        text_dim = getattr(mc, 'text_dim', None) or global_configs.TEXT_DIM
        acoustic_dim = getattr(mc, 'acoustic_dim', None) or global_configs.ACOUSTIC_DIM
        visual_dim = getattr(mc, 'visual_dim', None) or global_configs.VISUAL_DIM

    if text_dim <= 0:
        text_dim = config.hidden_size
    if acoustic_dim <= 0 or visual_dim <= 0:
        raise ValueError(
            "Acoustic / visual dims not configured. "
            "Call global_configs.set_dataset_config(...) before model creation."
        )
    return text_dim, acoustic_dim, visual_dim

class InfoGate_BertModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = _resolve_dims(config, multimodal_config)
        self.config = config

        bert_dir = _resolve_bert_pretrained_dir(multimodal_config)
        model = BertModel.from_pretrained(bert_dir)
        self.model = model.to(DEVICE)

        ig_args = {
            'text_dim': TEXT_DIM,
            'acoustic_dim': ACOUSTIC_DIM,
            'visual_dim': VISUAL_DIM,
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
            'confidence_mode': getattr(multimodal_config, 'confidence_mode', 'token'),
            'routing_mode': getattr(multimodal_config, 'routing_mode', 'gumbel'),
            'task_type': getattr(multimodal_config, 'task_type', 'regression'),
            'ablation': getattr(multimodal_config, 'ablation', 'none'),
        }

        _abl = ig_args.get('ablation', 'none')
        if _abl == 'none':
            ig_fixed = {k: v for k, v in ig_args.items() if k != 'ablation'}
            self.infogate = InfoGateFixed(ig_fixed)
        else:
            self.infogate = InfoGatePrism(ig_args)
        self.init_weights()

    def forward(self, input_ids, visual, acoustic,
                labels=None, stage=1,
                attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
            attention_mask = input_ids.ne(pad_id).long()
        else:
            attention_mask = attention_mask.long()

        bert_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            bert_kwargs["token_type_ids"] = token_type_ids.long()

        text_features = self.model(**bert_kwargs)[0]  # [B, T, 768]

        logits, ib_loss, loss_dict, h_p = self.infogate(
            text_features, acoustic, visual,
            labels=labels, stage=stage,
            attention_mask=attention_mask,
        )
        return logits, ib_loss, loss_dict, h_p


class InfoGate_BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.bert = InfoGate_BertModel(config, multimodal_config)

    def forward(self, input_ids, visual, acoustic,
                labels=None, stage=1,
                attention_mask=None, token_type_ids=None):
        return self.bert(
            input_ids, visual, acoustic,
            labels=labels, stage=stage,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
