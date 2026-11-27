#!/usr/bin/env python3
"""Export per-sample DPR primary modality + soft routing weights (eval, deterministic).

``logs/optuna/4090D_restart/*.log`` only contain epoch-level ``Diag`` batch means;
they cannot identify which sample is A/V primary.  This script loads a checkpoint,
runs ``model.eval()`` on dev or test, and writes a CSV with:

  ``global_index``, ``split_index``, ``label``, ``logit``, ``abs_err``,
  ``route_entropy``, ``h_p_l2norm``,
  ``primary_idx``, ``primary_name``,
  ``w_a``, ``w_l``, ``w_v``, ``w_max``,
  ``ib_conf_t``, ``ib_conf_a``, ``ib_conf_v``, ``ib_conf_fused``

where ``w_*`` are the DPR ``softmax(logits)`` weights (see ``MSelector._route`` in
``infogate_modules.py``) and ``w_max = max(w_a,w_l,w_v)`` (routing confidence for
the chosen primary).  **Non-language-dominated** samples are those with
``primary_idx != 1`` (0=acoustic, 1=language, 2=visual).

``route_entropy`` is ``-sum_k w_k log w_k`` (nats).  ``h_p_l2norm`` is the L2 norm of
the primary pooled vector ``h_p`` (``InfoGate`` forward's fourth return) before the head.

``ib_conf_*`` are **VTB / IBEncoder** confidences ``sigmoid(-logvar)``, aggregated
per sample: mask-mean over valid tokens, then mean over bottleneck dimensions
(see ``ib_conf_utils.per_sample_ib_conf_mean``).  ``ib_conf_t`` aligns with
training logs ``conf_t`` (text slot); ``ib_conf_fused`` matches the spirit of
``fusion_conf`` on the **primary mixed** ``conf_p`` after DPR routing (per-sample
masked mean, not an identical scalar to the epoch log batch mean).

Canonical checkpoint path hints (relative to ``My_creation/``) live in
``Qualitative_Evaluation/GOLD_CHECKPOINT_PATHS.txt``.

Usage (from ``My_creation/``); CSV 建议写到 ``Qualitative_Evaluation/results/``::

    python Qualitative_Evaluation/dump_dpr_primary_on_split.py \\
      --checkpoint logs/optuna/4090D_restart/phase1/checkpoints/optuna_mosei_phase1/trial_70/infogate_mosei_best.pt \\
      --dataset mosei --split test \\
      --output Qualitative_Evaluation/results/mosei_trial70_dpr_test.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import DebertaV2Tokenizer

_QE_DIR = os.path.dirname(os.path.abspath(__file__))
_MY = os.path.dirname(_QE_DIR)
if _MY not in sys.path:
    sys.path.insert(0, _MY)
if _QE_DIR not in sys.path:
    sys.path.insert(0, _QE_DIR)

import global_configs  # noqa: E402
from deberta_infogate import InfoGate_DeBertaForSequenceClassification  # noqa: E402
from global_configs import DEVICE  # noqa: E402

from ib_conf_utils import (  # noqa: E402
    install_infogate_ib_trace,
    per_sample_dpr_entropy,
    per_sample_ib_conf_mean,
)
from simsv2_qual_utils import (  # noqa: E402
    apply_ckpt_arch_simsv2,
    build_simsv2_loader,
    default_simsv2_cli,
    forward_simsv2,
    load_simsv2_model,
    simsv2_infogate,
    unpack_simsv2_batch,
)


def _apply_ckpt_arch(cli: SimpleNamespace, ckpt_path: str) -> None:
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
        "ablation",
    )
    for k in keys:
        if hasattr(saved, k):
            setattr(cli, k, getattr(saved, k))
    for flag, attr in (
        ("disable_l_lib", "use_l_lib"),
        ("disable_l_rib", "use_l_rib"),
    ):
        if hasattr(saved, flag):
            setattr(cli, attr, not bool(getattr(saved, flag)))


class InputFeatures:
    __slots__ = ("input_ids", "visual", "acoustic", "input_mask", "segment_ids", "label_id")

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def prepare_deberta_input(tokens, visual, acoustic, tokenizer, max_seq_length, acoustic_dim, visual_dim):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    az = np.zeros((1, acoustic_dim))
    acoustic = np.concatenate((az, acoustic, az))
    vz = np.zeros((1, visual_dim))
    visual = np.concatenate((vz, visual, vz))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad = max_seq_length - len(input_ids)
    acoustic = np.concatenate((acoustic, np.zeros((pad, acoustic_dim))))
    visual = np.concatenate((visual, np.zeros((pad, visual_dim))))
    input_ids += [0] * pad
    input_mask += [0] * pad
    segment_ids += [0] * pad

    return input_ids, visual, acoustic, input_mask, segment_ids


def convert_to_features(examples, max_seq_length, tokenizer, dataset: str):
    global_configs.set_dataset_config(dataset)
    adim = global_configs.ACOUSTIC_DIM
    vdim = global_configs.VISUAL_DIM
    feats = []
    for example in examples:
        (words, visual, acoustic), label_id, _segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            toks = tokenizer.tokenize(word)
            tokens.extend(toks)
            inversions.extend([idx] * len(toks))

        aligned_v = np.array([visual[i] for i in inversions])
        aligned_a = np.array([acoustic[i] for i in inversions])

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            aligned_a = aligned_a[: max_seq_length - 2]
            aligned_v = aligned_v[: max_seq_length - 2]

        ids, vis, aud, mask, seg = prepare_deberta_input(
            tokens, aligned_v, aligned_a, tokenizer, max_seq_length, adim, vdim
        )
        feats.append(InputFeatures(ids, vis, aud, mask, seg, label_id))
    return feats


def build_tensor_dataset(feats):
    return TensorDataset(
        torch.tensor(np.array([f.input_ids for f in feats]), dtype=torch.long),
        torch.tensor(np.array([f.visual for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f.acoustic for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f.label_id for f in feats]), dtype=torch.float),
    )


def _weights_and_primary_from_mselector_output(out) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Parse ``MSelector`` forward return: 3-way (6-tuple) or 4-way (7-tuple)."""
    if not isinstance(out, tuple):
        return None, out if torch.is_tensor(out) else out[-1]
    if len(out) >= 6 and torch.is_tensor(out[3]) and out[3].dim() == 2:
        return out[3], out[5]
    if len(out) >= 7 and torch.is_tensor(out[4]) and out[4].dim() == 2:
        return out[4], out[6]
    return None, out[-1]


def _write_rows_and_summary(rows: list[dict], output: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "global_index", "split_index", "label", "logit", "abs_err",
        "route_entropy", "h_p_l2norm",
        "primary_idx", "primary_name",
        "w_a", "w_l", "w_v", "w_max",
        "ib_conf_t", "ib_conf_a", "ib_conf_v", "ib_conf_fused",
    ]
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    na = sum(1 for r in rows if r["primary_idx"] == 0)
    nl = sum(1 for r in rows if r["primary_idx"] == 1)
    nv = sum(1 for r in rows if r["primary_idx"] == 2)
    nnl = na + nv
    non_l = [r for r in rows if r["primary_idx"] != 1]
    wmax_non_l = [float(r["w_max"]) for r in non_l]
    mean_wmax_nl = sum(wmax_non_l) / len(wmax_non_l) if wmax_non_l else 0.0
    mean_abs = sum(float(r["abs_err"]) for r in rows) / max(n, 1)

    print(f"Wrote {n} rows to {output}")
    print(f"Counts: acoustic={na}  language={nl}  visual={nv}  non_language={nnl} ({100.0 * nnl / max(n,1):.2f}%)")
    print(f"Non-L DPR w_max mean: {mean_wmax_nl:.6f}  (n={len(non_l)})")
    print(f"Mean |logit-label| (subset MAE proxy): {mean_abs:.6f}")


def _run_simsv2(ckpt_path: str, split: str, output: str, batch_size: int) -> int:
    cli = default_simsv2_cli()
    apply_ckpt_arch_simsv2(cli, ckpt_path)
    try:
        dl, _split_key, _data_path = build_simsv2_loader(cli, split, batch_size)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not load SIMSv2 split: {exc}", file=sys.stderr)
        return 1

    model = load_simsv2_model(cli, ckpt_path)

    captured: list[tuple[torch.Tensor | None, torch.Tensor]] = []

    def _hook(_mod, _inp, out):
        w, p = _weights_and_primary_from_mselector_output(out)
        captured.append((w.detach().cpu() if w is not None else None, p.detach().cpu()))

    ig = simsv2_infogate(model)
    h = ig.mselector.register_forward_hook(_hook)
    cleanup_ib, ib_store = install_infogate_ib_trace(ig)

    names = {0: "acoustic", 1: "language", 2: "visual"}
    rows = []
    offset = 0
    try:
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"DPR {split}"):
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, visual, acoustic, labels, input_mask, segment_ids = unpack_simsv2_batch(batch)
                attn = input_mask.float()
                ib_store.clear()
                captured.clear()
                logits, _, _, h_p = forward_simsv2(
                    model, input_ids, visual, acoustic,
                    labels=None, input_mask=input_mask, segment_ids=segment_ids,
                )
                if not captured:
                    print("ERROR: mselector hook did not fire", file=sys.stderr)
                    return 1
                wcpu, pidx = captured[-1]
                if wcpu is None:
                    print("ERROR: could not parse DPR weights from mselector output", file=sys.stderr)
                    return 1
                if wcpu.size(-1) != 3:
                    print(f"ERROR: expected 3 DPR modalities for simsv2, got {wcpu.size(-1)}", file=sys.stderr)
                    return 1
                if not {"t", "a", "v", "conf_p"}.issubset(ib_store.keys()):
                    print(
                        "ERROR: IB conf hooks missing keys "
                        f"(have {sorted(ib_store.keys())})",
                        file=sys.stderr,
                    )
                    return 1
                ib_t = per_sample_ib_conf_mean(ib_store["t"], attn).cpu()
                ib_a = per_sample_ib_conf_mean(ib_store["a"], attn).cpu()
                ib_v = per_sample_ib_conf_mean(ib_store["v"], attn).cpu()
                ib_f = per_sample_ib_conf_mean(ib_store["conf_p"], attn).cpu()
                logit_b = logits.squeeze(-1).detach().cpu()
                y_b = labels.squeeze(-1).detach().cpu()
                hp_norm = h_p.squeeze(-1).norm(dim=-1).detach().cpu()
                ent_b = per_sample_dpr_entropy(wcpu).cpu()
                for j in range(pidx.size(0)):
                    pi = int(pidx[j].item())
                    wa = float(wcpu[j, 0].item())
                    wl = float(wcpu[j, 1].item())
                    wv = float(wcpu[j, 2].item())
                    wmax = max(wa, wl, wv)
                    yj = float(y_b[j].item())
                    lj = float(logit_b[j].item())
                    rows.append(
                        {
                            "global_index": offset + j,
                            "split_index": offset + j,
                            "label": yj,
                            "logit": lj,
                            "abs_err": abs(lj - yj),
                            "route_entropy": float(ent_b[j].item()),
                            "h_p_l2norm": float(hp_norm[j].item()),
                            "primary_idx": pi,
                            "primary_name": names.get(pi, str(pi)),
                            "w_a": wa,
                            "w_l": wl,
                            "w_v": wv,
                            "w_max": wmax,
                            "ib_conf_t": float(ib_t[j].item()),
                            "ib_conf_a": float(ib_a[j].item()),
                            "ib_conf_v": float(ib_v[j].item()),
                            "ib_conf_fused": float(ib_f[j].item()),
                        }
                    )
                offset += pidx.size(0)
    finally:
        h.remove()
        cleanup_ib()

    _write_rows_and_summary(rows, output)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, choices=("mosi", "mosei", "simsv2"), required=True)
    ap.add_argument("--split", type=str, choices=("train", "dev", "test"), default="test")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=128)
    args = ap.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    if args.dataset == "simsv2":
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        return _run_simsv2(ckpt_path, args.split, args.output, args.batch_size)

    cli = SimpleNamespace(
        model=os.path.join(_MY, "deberta-v3-base"),
        dataset=args.dataset,
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
        ablation="none",
        use_l_lib=True,
        use_l_rib=True,
    )
    _apply_ckpt_arch(cli, ckpt_path)
    if not getattr(cli, "ablation", None):
        cli.ablation = "none"

    global_configs.set_dataset_config(cli.dataset)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
        cli.model, multimodal_config=cli, num_labels=1
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    captured: list[tuple[torch.Tensor | None, torch.Tensor]] = []

    def _hook(_mod, _inp, out):
        w, p = _weights_and_primary_from_mselector_output(out)
        captured.append(
            (w.detach().cpu() if w is not None else None, p.detach().cpu())
        )

    ig = model.dberta.infogate
    h = ig.mselector.register_forward_hook(_hook)
    cleanup_ib, ib_store = install_infogate_ib_trace(ig)

    ds_path = os.path.join(_MY, "datasets", f"{cli.dataset}.pkl")
    with open(ds_path, "rb") as fh:
        data = pickle.load(fh)
    if args.split == "train":
        if "train" not in data:
            print("ERROR: no train split in pickle", file=sys.stderr)
            return 1
        raw = data["train"]
    elif args.split == "dev":
        key = "dev" if "dev" in data else ("valid" if "valid" in data else None)
        if key is None:
            print("ERROR: no dev/valid split in pickle", file=sys.stderr)
            return 1
        raw = data[key]
    else:
        raw = data["test"]

    tok = DebertaV2Tokenizer.from_pretrained(cli.model)
    feats = convert_to_features(raw, cli.max_seq_length, tok, cli.dataset)
    ds = build_tensor_dataset(feats)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    names = {0: "acoustic", 1: "language", 2: "visual"}
    rows = []
    offset = 0
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    try:
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"DPR {args.split}"):
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, visual, acoustic, labels = batch
                visual = visual.squeeze(1)
                acoustic = acoustic.squeeze(1)
                attn = input_ids.ne(pad_id).float()
                ib_store.clear()
                captured.clear()
                logits, _, _, h_p = model(input_ids, visual, acoustic, labels=None, stage=2)
                if not captured:
                    print("ERROR: mselector hook did not fire", file=sys.stderr)
                    return 1
                wcpu, pidx = captured[-1]
                if wcpu is None:
                    print("ERROR: could not parse DPR weights from mselector output", file=sys.stderr)
                    return 1
                if wcpu.size(-1) != 3:
                    print(
                        f"ERROR: expected 3 DPR modalities for mosi/mosei, got {wcpu.size(-1)}",
                        file=sys.stderr,
                    )
                    return 1
                if not {"t", "a", "v", "conf_p"}.issubset(ib_store.keys()):
                    print(
                        "ERROR: IB conf hooks missing keys "
                        f"(have {sorted(ib_store.keys())})",
                        file=sys.stderr,
                    )
                    return 1
                ib_t = per_sample_ib_conf_mean(ib_store["t"], attn).cpu()
                ib_a = per_sample_ib_conf_mean(ib_store["a"], attn).cpu()
                ib_v = per_sample_ib_conf_mean(ib_store["v"], attn).cpu()
                ib_f = per_sample_ib_conf_mean(ib_store["conf_p"], attn).cpu()
                logit_b = logits.squeeze(-1).detach().cpu()
                y_b = labels.squeeze(-1).detach().cpu()
                hp_norm = h_p.squeeze(-1).norm(dim=-1).detach().cpu()
                ent_b = per_sample_dpr_entropy(wcpu).cpu()
                for j in range(pidx.size(0)):
                    pi = int(pidx[j].item())
                    wa = float(wcpu[j, 0].item())
                    wl = float(wcpu[j, 1].item())
                    wv = float(wcpu[j, 2].item())
                    wmax = max(wa, wl, wv)
                    yj = float(y_b[j].item())
                    lj = float(logit_b[j].item())
                    rows.append(
                        {
                            "global_index": offset + j,
                            "split_index": offset + j,
                            "label": yj,
                            "logit": lj,
                            "abs_err": abs(lj - yj),
                            "route_entropy": float(ent_b[j].item()),
                            "h_p_l2norm": float(hp_norm[j].item()),
                            "primary_idx": pi,
                            "primary_name": names.get(pi, str(pi)),
                            "w_a": wa,
                            "w_l": wl,
                            "w_v": wv,
                            "w_max": wmax,
                            "ib_conf_t": float(ib_t[j].item()),
                            "ib_conf_a": float(ib_a[j].item()),
                            "ib_conf_v": float(ib_v[j].item()),
                            "ib_conf_fused": float(ib_f[j].item()),
                        }
                    )
                offset += pidx.size(0)
    finally:
        h.remove()
        cleanup_ib()

    _write_rows_and_summary(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
