#!/usr/bin/env python3
"""SIMSv2 qualitative-evaluation helpers (BERT + MMSA tensors)."""

from __future__ import annotations

import os
import pickle
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

_QE_DIR = os.path.dirname(os.path.abspath(__file__))
_MY = os.path.dirname(_QE_DIR)

import global_configs  # noqa: E402
from bert_infogate import InfoGate_BertForSequenceClassification  # noqa: E402
from global_configs import DEVICE  # noqa: E402
from simsv2_mmsa_data import build_tensor_dataset as build_simsv2_mmsa_dataset  # noqa: E402


def default_simsv2_cli() -> SimpleNamespace:
    return SimpleNamespace(
        model=os.path.join(_MY, "bert-base-chinese"),
        dataset="simsv2",
        max_seq_length=50,
        unified_dim=256,
        ib_hidden_dim=256,
        bottleneck_dim=128,
        num_heads=4,
        num_infogate_layers=3,
        dropout_prob=0.1,
        beta_ib=32.0,
        alpha_ib=0.01,
        mse_weight=0.5,
        selector_target_temp=0.35,
        selector_balance_weight=0.0,
        selector_rib_weight=0.05,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.5,
        simsv2_feature_mode="mmsa",
        ablation="none",
        use_l_lib=True,
        use_l_rib=True,
    )


def apply_ckpt_arch_simsv2(cli: SimpleNamespace, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args")
    if saved is None:
        return
    keys = (
        "model",
        "dataset",
        "max_seq_length",
        "unified_dim",
        "ib_hidden_dim",
        "bottleneck_dim",
        "num_heads",
        "num_infogate_layers",
        "dropout_prob",
        "beta_ib",
        "alpha_ib",
        "mse_weight",
        "selector_target_temp",
        "selector_balance_weight",
        "selector_rib_weight",
        "gumbel_tau_start",
        "gumbel_tau_end",
        "simsv2_feature_mode",
        "ablation",
    )
    for key in keys:
        if hasattr(saved, key):
            setattr(cli, key, getattr(saved, key))
    for flag, attr in (
        ("disable_l_lib", "use_l_lib"),
        ("disable_l_rib", "use_l_rib"),
    ):
        if hasattr(saved, flag):
            setattr(cli, attr, not bool(getattr(saved, flag)))
    if not getattr(cli, "model", "") or "deberta-v3-base" in str(cli.model):
        cli.model = os.path.join(_MY, "bert-base-chinese")
    cli.dataset = "simsv2"
    cli.simsv2_feature_mode = "mmsa"


def load_simsv2_model(cli: SimpleNamespace, ckpt_path: str):
    global_configs.set_dataset_config("simsv2")
    model = InfoGate_BertForSequenceClassification.from_pretrained(
        cli.model, multimodal_config=cli, num_labels=1
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def simsv2_infogate(model):
    return model.bert.infogate


def simsv2_split_key(data: dict, split: str) -> str:
    if split == "train":
        if "train" not in data:
            raise KeyError("SIMSv2 pickle has no train split")
        return "train"
    if split == "dev":
        if "dev" in data:
            return "dev"
        if "valid" in data:
            return "valid"
        raise KeyError("SIMSv2 pickle has no dev/valid split")
    return "test"


def load_simsv2_split(split: str):
    data_path = os.path.join(_MY, "datasets", "simsv2.pkl")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"SIMSv2 dataset not found: {data_path}. "
            "Run/restore My_creation/datasets/simsv2.pkl before qualitative dumps."
        )
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    key = simsv2_split_key(data, split)
    return data[key], key, data_path


def build_simsv2_loader(cli: SimpleNamespace, split: str, batch_size: int):
    split_data, split_key, data_path = load_simsv2_split(split)
    dataset = build_simsv2_mmsa_dataset(split_data, int(cli.max_seq_length))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), split_key, data_path


def unpack_simsv2_batch(batch):
    input_ids, visual, acoustic, input_mask, segment_ids, labels = batch[:6]
    return input_ids, visual, acoustic, labels, input_mask, segment_ids


def forward_simsv2(model, input_ids, visual, acoustic, labels=None, input_mask=None, segment_ids=None):
    return model(
        input_ids,
        visual.squeeze(1),
        acoustic.squeeze(1),
        labels=labels,
        stage=2,
        attention_mask=input_mask,
        token_type_ids=segment_ids,
    )


def load_single_simsv2_batch(cli: SimpleNamespace, split: str, global_index: int, batch_size: int = 64):
    loader, split_key, _data_path = build_simsv2_loader(cli, split, batch_size)
    n = len(loader.dataset)
    if global_index < 0 or global_index >= n:
        raise IndexError(f"global_index {global_index} out of range [0,{n - 1}] for {split_key}")
    batch_idx = global_index // batch_size
    inner = global_index % batch_size
    for idx, batch in enumerate(loader):
        if idx == batch_idx:
            return tuple(t.to(DEVICE) for t in batch), inner
    raise RuntimeError("SIMSv2 batch not found")
