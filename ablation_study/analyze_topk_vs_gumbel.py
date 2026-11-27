#!/usr/bin/env python3
"""Gumbel primary vs top-2 assignment, plus per-modality VTB confidence.

Top-k is the existing ``routing_mode=topk``: the two highest soft scores share
the primary stream equally, the mask is detached, and no Gumbel noise is added.
Gumbel is the paper router: one hard primary, straight-through in training and
argmax at evaluation. ``no_ib`` keeps that Gumbel router but replaces VTB with a
linear map and confidence 1, so a change in who is selected is due to the
bottleneck representation rather than the confidence values.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MY = _HERE.parent
sys.path.insert(0, str(_HERE))
import tsne_sota3090_more as tsne  # noqa: E402

SOTA = _MY / "fixed_experiment" / "runs" / "sota_2026_11"
OUT = _HERE / "runs" / "topk_vs_gumbel"

CASES = (
    {
        "key": "simsv2_trial159",
        "title": "CH-SIMS v2",
        "loader": "regression",
        "script": "train.py",
        "trial": "simsv2_trial159",
        "ckpt_name": "infogate_simsv2_best.pt",
        "names": ("acoustic", "language", "visual"),
        "metric": "mae",
    },
    {
        "key": "mustard_trial1239",
        "title": "MUStARD",
        "loader": "classify",
        "script": "train_classify.py",
        "trial": "mustard_trial1239",
        "ckpt_name": "infogate_mustard_best.pt",
        "names": ("acoustic", "language", "visual", "hcf"),
        "metric": "acc",
    },
)


def _ckpt(key: str, variant: str, ckpt_name: str) -> Path:
    folder = key if variant == "gumbel" else f"{key}_{variant if variant != 'topk' else 'route_topk'}"
    if variant == "no_ib":
        folder = f"{key}_no_ib"
    return SOTA / folder / "checkpoints" / ckpt_name


def _log(key: str, variant: str) -> Path:
    folder = key if variant == "gumbel" else (
        f"{key}_no_ib" if variant == "no_ib" else f"{key}_route_topk"
    )
    return SOTA / folder / "train.log"


def _argv(case: dict, variant: str, ckpt: Path) -> list[str]:
    sys.path.insert(0, str(_MY / "fixed_experiment"))
    from sota_2026_11_trials import build_argv
    argv = build_argv(case["trial"], str(ckpt.parent))
    if variant == "no_ib":
        argv += ["--ablation", "no_ib"]
    elif variant == "topk":
        argv += ["--routing_mode", "topk"]
    return argv


def _find_infogate(model):
    for mod in model.modules():
        if mod.__class__.__name__ == "InfoGate" and hasattr(mod, "mselector"):
            return mod
    raise RuntimeError("InfoGate with a router was not found")


def _collect(case: dict, variant: str, ckpt: Path):
    mod = tsne._load_module(case["script"], _argv(case, variant, ckpt), "_topk_train")
    mod.set_seed(int(mod.args.seed))
    _, _, test_dl, n_opt = mod.setup_data()
    model, _, _ = mod.build_model(n_opt)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.to(mod.DEVICE)
    model.eval()
    ig = _find_infogate(model)
    box: dict = {}

    def _capture(output):
        # weights, primary_idx are the last-but-two and the last return values
        # of both MSelector.forward and MSelector.forward4. forward4 is invoked
        # directly, so a module hook on __call__ would miss the 4-modality path.
        box["weights"] = output[-3].detach().float().cpu()
        box["idx"] = output[-1].detach().cpu()
        return output

    orig_forward = ig.mselector.forward
    orig_forward4 = ig.mselector.forward4

    def wrapped_forward(*args, **kwargs):
        return _capture(orig_forward(*args, **kwargs))

    def wrapped_forward4(*args, **kwargs):
        return _capture(orig_forward4(*args, **kwargs))

    ig.mselector.forward = wrapped_forward
    ig.mselector.forward4 = wrapped_forward4
    for slot, enc_name in (("a", "ib_enc_a"), ("t", "ib_enc_t"), ("v", "ib_enc_v"), ("h", "ib_enc_h")):
        enc = getattr(ig, enc_name, None)
        if enc is None:
            continue

        def _conf_hook(_module, _inputs, output, slot=slot):
            box[f"conf_{slot}"] = output[3].detach().float().mean(dim=(1, 2)).cpu()

        enc.register_forward_hook(_conf_hook)

    weights, idx, labels, preds = [], [], [], []
    confs = {slot: [] for slot in ("a", "t", "v", "h")}
    use_hcf = case["loader"] == "classify" and mod.effective_hcf_dim > 0
    with torch.no_grad():
        for batch in test_dl:
            batch = tuple(t.to(mod.DEVICE) for t in batch)
            box.clear()
            if case["loader"] == "classify":
                input_ids, visual, acoustic, hcf, label_ids = mod._unpack_batch(batch, use_hcf)
                logits, _, _, _ = model(input_ids, visual, acoustic, hcf=hcf, stage=2)
            else:
                use_mmsa = case["key"].startswith("simsv2") and mod.simsv2_mmsa_enabled()
                input_ids, visual, acoustic, label_ids, input_mask, segment_ids = (
                    mod.unpack_simsv2_batch(batch, use_mmsa)
                )
                visual = visual.squeeze(1)
                acoustic = acoustic.squeeze(1)
                logits, _, _, _ = mod._forward_model(
                    model, input_ids, visual, acoustic, label_ids, 2, input_mask, segment_ids,
                )
            weights.append(box["weights"].numpy())
            idx.append(box["idx"].numpy())
            labels.append(label_ids.detach().cpu().numpy().reshape(-1))
            preds.append(logits.detach().float().cpu().numpy().reshape(-1))
            for slot in confs:
                if f"conf_{slot}" in box:
                    confs[slot].append(box[f"conf_{slot}"].numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = {
        "weights": np.concatenate(weights, 0),
        "idx": np.concatenate(idx, 0).astype(np.int64),
        "y": np.concatenate(labels, 0).astype(np.float64),
        "pred": np.concatenate(preds, 0).astype(np.float64),
    }
    for slot, parts in confs.items():
        if parts:
            out[f"conf_{slot}"] = np.concatenate(parts, 0)
    return out


def _top2(weights: np.ndarray) -> np.ndarray:
    k = min(2, weights.shape[1])
    picked = np.argsort(-weights, axis=1)[:, :k]
    mask = np.zeros_like(weights, dtype=bool)
    rows = np.arange(weights.shape[0])[:, None]
    mask[rows, picked] = True
    return mask


def _metric(pred, y, metric: str) -> float:
    if len(y) == 0:
        return float("nan")
    if metric == "mae":
        return float(np.mean(np.abs(pred - y)))
    prob = 1.0 / (1.0 + np.exp(-pred))
    return float(np.mean((prob >= 0.5).astype(np.int32) == y.astype(np.int32)))


def _summarize(case: dict, dumps: dict[str, dict]) -> list[str]:
    names = case["names"]
    lines = [f"## {case['title']}  test n={len(next(iter(dumps.values()))['y'])}"]
    gumbel = dumps.get("gumbel")
    topk = dumps.get("topk")
    no_ib = dumps.get("no_ib")
    for name, dump in dumps.items():
        n = len(dump["y"])
        rates = [(dump["idx"] == i).mean() for i in range(len(names))]
        rate_s = " ".join(f"{names[i]} {rates[i]*100:.1f}%" for i in range(len(names)))
        lines.append(f"{name} hard-top1: {rate_s}")
        mask = _top2(dump["weights"])
        top_s = " ".join(f"{names[i]} {mask[:, i].mean()*100:.1f}%" for i in range(len(names)))
        lines.append(f"{name} in-top2: {top_s}")
        if name == "topk":
            pairs = {}
            for row in mask:
                key = tuple(names[i] for i in range(len(names)) if row[i])
                pairs[key] = pairs.get(key, 0) + 1
            ranked = sorted(pairs.items(), key=lambda kv: -kv[1])
            lines.append("topk pairs: " + ", ".join(
                f"{'+'.join(k)} {v} ({v/n*100:.1f}%)" for k, v in ranked))
        overall = _metric(dump["pred"], dump["y"], case["metric"])
        lines.append(f"{name} test {case['metric']} {overall:.4f}")
        for i, modality in enumerate(names):
            if name == "topk":
                chosen = mask[:, i]
            else:
                chosen = dump["idx"] == i
            m = _metric(dump["pred"][chosen], dump["y"][chosen], case["metric"])
            extra = ""
            if case["metric"] == "mae" and chosen.any():
                extra = f" |y| {np.abs(dump['y'][chosen]).mean():.3f}"
            lines.append(f"  {modality} selected n={int(chosen.sum())} {case['metric']} {m:.4f}{extra}")
        slots = ("a", "t", "v", "h")[:len(names)]
        conf_bits = []
        for slot, modality in zip(slots, names):
            col = dump.get(f"conf_{slot}")
            if col is None:
                conf_bits.append(f"{modality} conf=1 (no VTB)")
                continue
            chosen = mask[:, slots.index(slot)] if name == "topk" else (dump["idx"] == slots.index(slot))
            # slots.index is wrong if I used slots from the slice. Use enumerate.
            conf_bits.append(f"{modality} {col.mean():.3f}")
        lines.append(name + " mean VTB conf: " + " ".join(conf_bits))
        for i, (slot, modality) in enumerate(zip(slots, names)):
            col = dump.get(f"conf_{slot}")
            if col is None:
                continue
            chosen = mask[:, i] if name == "topk" else (dump["idx"] == i)
            if chosen.any() and (~chosen).any():
                lines.append(
                    f"  {modality} conf selected {col[chosen].mean():.3f} "
                    f"vs not {col[~chosen].mean():.3f}"
                )
    if gumbel is not None and topk is not None and len(gumbel["idx"]) == len(topk["idx"]):
        mask = _top2(topk["weights"])
        inside = mask[np.arange(len(gumbel["idx"])), gumbel["idx"]].mean()
        same_top1 = (gumbel["idx"] == topk["idx"]).mean()
        lines.append(
            f"gumbel top-1 is inside that sample's top-k set on {inside*100:.1f}% of test samples; "
            f"the two models pick the same top-1 on {same_top1*100:.1f}%"
        )
    if gumbel is not None and no_ib is not None:
        same = (gumbel["idx"] == no_ib["idx"]).mean()
        lines.append(
            f"removing VTB changes the hard primary on {(1-same)*100:.1f}% of test samples "
            f"(same primary on {same*100:.1f}%)"
        )
    return lines


_PAIR = re.compile(r"([alhvt]):([0-9.]+)")


def _pairs(blob: str) -> dict[str, float]:
    return {k: float(v) for k, v in _PAIR.findall(blob)}


def _epoch_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    epoch = None
    primary, top2, conf, weights = {}, {}, {}, {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("Epoch "):
            if epoch is not None and primary:
                rows.append({
                    "epoch": epoch, "primary": primary, "top2": top2,
                    "conf": conf, "weights": weights,
                })
            epoch = int(line.split()[1].split("/")[0])
            primary, top2, conf, weights = {}, {}, {}, {}
            continue
        if "primary=[" in line:
            primary = _pairs(line.split("primary=[", 1)[1].split("]", 1)[0])
        if "top2=[" in line:
            top2 = _pairs(line.split("top2=[", 1)[1].split("]", 1)[0])
        if "conf=[" in line:
            conf = _pairs(line.split("conf=[", 1)[1].split("]", 1)[0])
        if "w=[" in line:
            weights = _pairs(line.split("w=[", 1)[1].split("]", 1)[0])
    if epoch is not None and primary:
        rows.append({"epoch": epoch, "primary": primary, "top2": top2, "conf": conf, "weights": weights})
    return rows


def _plot_epochs(case: dict, traces: dict[str, list[dict]]) -> None:
    if not any(traces.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    colors = {"a": "#20B2AA", "l": "#ae017e", "v": "#0066CC", "h": "#E69F00"}
    labels = {"a": "acoustic", "l": "language", "v": "visual", "h": "hcf"}
    for ax, (title, field) in zip(axes, (("hard top-1", "primary"), ("in top-2", "top2"))):
        for variant, rows in traces.items():
            if not rows or not rows[-1][field]:
                continue
            for key in rows[-1][field]:
                ys = [row[field].get(key, np.nan) for row in rows]
                xs = [row["epoch"] for row in rows]
                ax.plot(xs, ys, color=colors.get(key, "black"),
                        linestyle={"gumbel": "-", "topk": "--", "no_ib": ":"}.get(variant, "-"),
                        linewidth=1.2, label=f"{variant} {labels.get(key, key)}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel("fraction of training batch")
    handles, labs = axes[1].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(case["title"] + "  routing over training", fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{case['key']}_routing_over_training.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=tuple(c["key"] for c in CASES), default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os_lines = []
    os.chdir(_MY)
    selected = [c for c in CASES if args.case is None or c["key"] == args.case]
    for case in selected:
        print(f"=== {case['key']} ===", flush=True)
        dumps = {}
        for variant in ("gumbel", "topk", "no_ib"):
            ckpt = _ckpt(case["key"], variant, case["ckpt_name"])
            if not ckpt.is_file():
                print(f"  missing {variant}: {ckpt}", flush=True)
                continue
            print(f"  load {variant}", flush=True)
            dumps[variant] = _collect(case, variant, ckpt)
            np.savez(OUT / f"{case['key']}_{variant}.npz", **dumps[variant])
        if not dumps:
            continue
        lines = _summarize(case, dumps)
        os_lines.extend(lines)
        os_lines.append("")
        for line in lines:
            print(line, flush=True)
        case_report = OUT / f"{case['key']}_report.txt"
        case_report.write_text("\n".join(lines) + "\n")
        traces = {variant: _epoch_rows(_log(case["key"], variant)) for variant in ("gumbel", "topk", "no_ib")}
        _plot_epochs(case, traces)
    report = OUT / "report.txt"
    report.write_text("\n".join(os_lines) + "\n")
    print(f"wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
