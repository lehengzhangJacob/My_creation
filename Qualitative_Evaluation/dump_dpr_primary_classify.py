#!/usr/bin/env python3
"""Export per-sample DPR routing for ALBERT binary tasks (MUStARD, UR-FUNNY).

Same semantics as ``dump_dpr_primary_on_split.py`` (DeBERTa regression), but
uses ``InfoGate_AlbertForSequenceClassification`` + ``data_humor.build_humor_loaders``
and records ``MSelector`` outputs.  With HCF (MUStARD / UR-FUNNY), InfoGate
calls ``forward4`` instead of ``forward``, so this script wraps ``forward4``
when ``num_modalities == 4`` (otherwise a forward-hook on ``mselector``).

Output CSV columns:

- DPR: ``w_a``, ``w_l``, ``w_v`` (``w_hcf`` in 4-way HCF), ``w_max``.
- Task: ``label``, ``logit``, ``prob`` (sigmoid), ``pred_bin`` (threshold 0.5),
  ``pred_correct``, ``route_entropy``, ``h_p_l2norm``.
- VTB IB conf: ``ib_conf_t``, ``ib_conf_a``, ``ib_conf_v``, optional ``ib_conf_h``,
  ``ib_conf_fused`` (primary mixed ``conf_p``; see ``ib_conf_utils``).

4-way (HCF): slot order is (acoustic, language, visual, hcf).
Non-language-dominated: ``primary_idx != 1``.

Usage (from ``My_creation/``)::

    python Qualitative_Evaluation/dump_dpr_primary_classify.py \\
      --checkpoint logs/optuna/.../infogate_mustard_best.pt \\
      --dataset mustard --split test \\
      --output Qualitative_Evaluation/results/mustard_dpr_test.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from transformers import AlbertTokenizer

_QE_DIR = os.path.dirname(os.path.abspath(__file__))
_MY = os.path.dirname(_QE_DIR)
if _MY not in sys.path:
    sys.path.insert(0, _MY)

import global_configs  # noqa: E402
from albert_infogate import InfoGate_AlbertForSequenceClassification  # noqa: E402
from data_humor import build_humor_loaders  # noqa: E402
from global_configs import DEVICE  # noqa: E402

from ib_conf_utils import (  # noqa: E402
    install_infogate_ib_trace,
    per_sample_dpr_entropy,
    per_sample_ib_conf_mean,
)


def _weights_and_primary_from_mselector_output(out) -> tuple[torch.Tensor | None, torch.Tensor]:
    if not isinstance(out, tuple):
        return None, out if torch.is_tensor(out) else out[-1]
    if len(out) >= 6 and torch.is_tensor(out[3]) and out[3].dim() == 2:
        return out[3], out[5]
    if len(out) >= 7 and torch.is_tensor(out[4]) and out[4].dim() == 2:
        return out[4], out[6]
    return None, out[-1]


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
        "task_type",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, choices=("mustard", "ur_funny"), required=True)
    ap.add_argument("--split", type=str, choices=("train", "dev", "test"), default="test")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument(
        "--model",
        type=str,
        default=os.path.join(_MY, "albert-base-v2"),
        help="ALBERT backbone directory (overridden by checkpoint args if present).",
    )
    args = ap.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    cli = SimpleNamespace(
        model=args.model,
        dataset=args.dataset,
        max_seq_length=64,
        unified_dim=256,
        ib_hidden_dim=256,
        bottleneck_dim=128,
        num_heads=4,
        num_infogate_layers=3,
        dropout_prob=0.25,
        beta_ib=16.0,
        alpha_ib=0.005,
        mse_weight=0.0,
        selector_target_temp=0.6,
        selector_balance_weight=0.0,
        selector_rib_weight=0.05,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.5,
        task_type="binary",
        use_l_lib=True,
        use_l_rib=True,
    )
    _apply_ckpt_arch(cli, ckpt_path)
    if not getattr(cli, "task_type", None):
        cli.task_type = "binary"
    if not hasattr(cli, "use_l_lib"):
        cli.use_l_lib = True
    if not hasattr(cli, "use_l_rib"):
        cli.use_l_rib = True
    if not hasattr(cli, "mse_weight"):
        cli.mse_weight = 0.0

    global_configs.set_dataset_config(cli.dataset)
    hcf_dim = global_configs.HCF_DIM
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = InfoGate_AlbertForSequenceClassification.from_pretrained(
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

    msel = model.albert.infogate.mselector
    n_mod = int(msel.num_modalities)
    hook_handle = None
    _orig_forward4 = None
    if n_mod == 3:
        hook_handle = msel.register_forward_hook(_hook)
    elif n_mod == 4:
        _orig_forward4 = msel.forward4

        def _forward4_capture(H_a, H_l, H_v, H_hcf, mask=None):
            out = _orig_forward4(H_a, H_l, H_v, H_hcf, mask)
            captured.append((out[4].detach().cpu(), out[6].detach().cpu()))
            return out

        msel.forward4 = _forward4_capture
    else:
        print(f"ERROR: unsupported MSelector.num_modalities={n_mod}", file=sys.stderr)
        return 1

    ig = model.albert.infogate
    cleanup_ib, ib_store = install_infogate_ib_trace(ig)

    tok = AlbertTokenizer.from_pretrained(cli.model)
    _tr, dev_dl, test_dl, _nopt = build_humor_loaders(
        dataset=cli.dataset,
        tokenizer=tok,
        max_seq_length=cli.max_seq_length,
        acoustic_dim=global_configs.ACOUSTIC_DIM,
        visual_dim=global_configs.VISUAL_DIM,
        train_batch_size=args.batch_size,
        dev_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        gradient_accumulation_step=1,
        n_epochs=1,
        hcf_dim=hcf_dim,
        slice_hkt=True,
    )
    if args.split == "train":
        dl = _tr
    elif args.split == "dev":
        dl = dev_dl
    else:
        dl = test_dl
    del _nopt

    names_3 = {0: "acoustic", 1: "language", 2: "visual"}
    names_4 = {0: "acoustic", 1: "language", 2: "visual", 3: "hcf"}
    rows = []
    offset = 0
    use_hcf = hcf_dim > 0
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    try:
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"DPR classify {args.split}"):
                batch = tuple(t.to(DEVICE) for t in batch)
                if use_hcf:
                    input_ids, visual, acoustic, hcf, labels = batch
                    hcf = hcf.squeeze(1)
                else:
                    input_ids, visual, acoustic, labels = batch
                    hcf = None
                visual = visual.squeeze(1)
                acoustic = acoustic.squeeze(1)
                attn = input_ids.ne(pad_id).float()
                ib_store.clear()
                captured.clear()
                logits, _, _, h_p = model(input_ids, visual, acoustic, hcf=hcf, stage=2)
                if not captured:
                    print("ERROR: mselector did not emit routing (hook/wrapper)", file=sys.stderr)
                    return 1
                wcpu, pidx = captured[-1]
                if wcpu is None:
                    print("ERROR: could not parse DPR weights from mselector output", file=sys.stderr)
                    return 1
                n_w = wcpu.size(-1)
                if n_w not in (3, 4):
                    print(f"ERROR: expected 3 or 4 DPR logits, got {n_w}", file=sys.stderr)
                    return 1
                need = {"t", "a", "v", "conf_p"}
                if use_hcf:
                    need.add("h")
                if not need.issubset(ib_store.keys()):
                    print(
                        "ERROR: IB conf trace missing keys",
                        sorted(ib_store.keys()),
                        file=sys.stderr,
                    )
                    return 1
                ib_t = per_sample_ib_conf_mean(ib_store["t"], attn).cpu()
                ib_a = per_sample_ib_conf_mean(ib_store["a"], attn).cpu()
                ib_v = per_sample_ib_conf_mean(ib_store["v"], attn).cpu()
                ib_f = per_sample_ib_conf_mean(ib_store["conf_p"], attn).cpu()
                ib_h_cpu = (
                    per_sample_ib_conf_mean(ib_store["h"], attn).cpu() if use_hcf else None
                )
                logit_b = logits.squeeze(-1).detach().cpu()
                y_b = labels.long().squeeze(-1).cpu()
                prob_b = torch.sigmoid(logits.squeeze(-1)).detach().cpu()
                pred_b = (prob_b >= 0.5).long()
                cor_b = (pred_b == y_b).float()
                hp_norm = h_p.squeeze(-1).norm(dim=-1).detach().cpu()
                ent_b = per_sample_dpr_entropy(wcpu).cpu()
                names = names_4 if n_w == 4 else names_3
                for j in range(pidx.size(0)):
                    pi = int(pidx[j].item())
                    wa = float(wcpu[j, 0].item())
                    wl = float(wcpu[j, 1].item())
                    wv = float(wcpu[j, 2].item())
                    task = {
                        "label": int(y_b[j].item()),
                        "logit": float(logit_b[j].item()),
                        "prob": float(prob_b[j].item()),
                        "pred_bin": int(pred_b[j].item()),
                        "pred_correct": float(cor_b[j].item()),
                        "route_entropy": float(ent_b[j].item()),
                        "h_p_l2norm": float(hp_norm[j].item()),
                    }
                    ib_row = {
                        "ib_conf_t": float(ib_t[j].item()),
                        "ib_conf_a": float(ib_a[j].item()),
                        "ib_conf_v": float(ib_v[j].item()),
                        "ib_conf_fused": float(ib_f[j].item()),
                    }
                    if n_w == 4:
                        wh = float(wcpu[j, 3].item())
                        wmax = max(wa, wl, wv, wh)
                        ib_row["ib_conf_h"] = float(ib_h_cpu[j].item()) if ib_h_cpu is not None else 0.0
                        rows.append(
                            {
                                "global_index": offset + j,
                                "split_index": offset + j,
                                **task,
                                "primary_idx": pi,
                                "primary_name": names.get(pi, str(pi)),
                                "w_a": wa,
                                "w_l": wl,
                                "w_v": wv,
                                "w_hcf": wh,
                                "w_max": wmax,
                                **ib_row,
                            }
                        )
                    else:
                        wmax = max(wa, wl, wv)
                        rows.append(
                            {
                                "global_index": offset + j,
                                "split_index": offset + j,
                                **task,
                                "primary_idx": pi,
                                "primary_name": names.get(pi, str(pi)),
                                "w_a": wa,
                                "w_l": wl,
                                "w_v": wv,
                                "w_max": wmax,
                                **ib_row,
                            }
                        )
                offset += pidx.size(0)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        elif _orig_forward4 is not None:
            msel.forward4 = _orig_forward4
        cleanup_ib()

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "global_index",
        "split_index",
        "label",
        "logit",
        "prob",
        "pred_bin",
        "pred_correct",
        "route_entropy",
        "h_p_l2norm",
        "primary_idx",
        "primary_name",
        "w_a",
        "w_l",
        "w_v",
    ]
    if use_hcf:
        fieldnames.append("w_hcf")
    fieldnames.extend(
        ["w_max", "ib_conf_t", "ib_conf_a", "ib_conf_v"]
    )
    if use_hcf:
        fieldnames.append("ib_conf_h")
    fieldnames.append("ib_conf_fused")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    na = sum(1 for r in rows if r["primary_idx"] == 0)
    nl = sum(1 for r in rows if r["primary_idx"] == 1)
    nv = sum(1 for r in rows if r["primary_idx"] == 2)
    nh = sum(1 for r in rows if r["primary_idx"] == 3)
    nnl = na + nv + nh
    non_l = [r for r in rows if r["primary_idx"] != 1]
    wmax_non_l = [float(r["w_max"]) for r in non_l]
    mean_wmax_nl = sum(wmax_non_l) / len(wmax_non_l) if wmax_non_l else 0.0

    mean_acc = sum(float(r["pred_correct"]) for r in rows) / max(n, 1)

    print(f"Wrote {n} rows to {args.output}")
    if nh > 0 or (rows and rows[0].get("w_hcf") is not None):
        print(
            f"Counts: acoustic={na}  language={nl}  visual={nv}  hcf={nh}  "
            f"non_language={nnl} ({100.0 * nnl / max(n, 1):.2f}%)"
        )
    else:
        print(
            f"Counts: acoustic={na}  language={nl}  visual={nv}  "
            f"non_language={nnl} ({100.0 * nnl / max(n, 1):.2f}%)"
        )
    print(f"Non-L DPR w_max mean: {mean_wmax_nl:.6f}  (n={len(non_l)})")
    print(f"Threshold-0.5 accuracy (pred vs label): {100.0 * mean_acc:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
