#!/usr/bin/env python3
"""t-SNE of test-set primary embeddings for 3090 SOTA ablations.

Each panel is an independent t-SNE of ``h_p`` (pooled primary, before the head).
Points are colored by true-label polarity, not by the model's prediction.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

_HERE = Path(__file__).resolve().parent
_MY = _HERE.parent
_HP = _MY / "Model_Complexity_Analysis"
OUT = _HERE / "runs" / "sota3090_tsne"

VARIANTS = (
    ("none", "PRISM (full)"),
    ("no_ib", "w/o VTB"),
    ("no_mselector", "w/o DPR"),
    ("no_infogate", "w/o InfoGate"),
)
CASES = (
    {
        "key": "mosei_deberta_trial36",
        "title": "CMU-MOSEI, DeBERTa",
        "hparams": _HP / "hparams_mosei_deberta_trial36.py",
        "root": _MY / "fixed_experiment/runs/sota_2026_11",
        "prefix": "mosei_deberta_trial36",
        "ckpt_name": "infogate_mosei_best.pt",
        "kind": "mosei",
    },
    {
        "key": "simsv2_trial159",
        "title": "CH-SIMS v2",
        "hparams": _HP / "hparams_simsv2_trial159.py",
        "root": _MY / "fixed_experiment/runs/sota_2026_11",
        "prefix": "simsv2_trial159",
        "ckpt_name": "infogate_simsv2_best.pt",
        "kind": "sims",
    },
)
COLORS = {"negative": "#20B2AA", "neutral": "#0066CC", "positive": "#ae017e"}
MARKERS = {"negative": "x", "neutral": "o", "positive": "+"}


def _load_build(path: Path):
    spec = importlib.util.spec_from_file_location(f"hp_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_train_argv


def _ckpt_dir(case: dict, ablation: str) -> Path:
    name = case["prefix"] if ablation == "none" else f"{case['prefix']}_{ablation}"
    return case["root"] / name / "checkpoints"


def _load_train(argv: list[str]):
    sys.argv = ["train.py", *argv]
    name = "_tsne3090_train"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, _HERE / "train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _collect(train_mod, ckpt: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    train_mod.set_seed(int(train_mod.args.seed))
    _, _, test_dl, n_opt = train_mod.setup_data()
    model, _, _ = train_mod.build_model(n_opt)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.to(train_mod.DEVICE)
    model.eval()
    hs, ys = [], []
    use_mmsa = kind == "sims" and train_mod.simsv2_mmsa_enabled()
    with torch.no_grad():
        for batch in test_dl:
            batch = tuple(t.to(train_mod.DEVICE) for t in batch)
            if kind == "sims":
                input_ids, visual, acoustic, label_ids, input_mask, segment_ids = (
                    train_mod.unpack_simsv2_batch(batch, use_mmsa)
                )
                visual = visual.squeeze(1)
                acoustic = acoustic.squeeze(1)
                _, _, _, h_p = train_mod._forward_model(
                    model, input_ids, visual, acoustic, label_ids, 2,
                    input_mask, segment_ids,
                )
            else:
                input_ids, visual, acoustic, label_ids = batch
                visual = visual.squeeze(1)
                acoustic = acoustic.squeeze(1)
                _, _, _, h_p = model(
                    input_ids, visual, acoustic, labels=label_ids, stage=2
                )
            hs.append(h_p.detach().float().cpu().numpy())
            ys.append(label_ids.detach().cpu().numpy().reshape(-1))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(hs, 0), np.concatenate(ys, 0)


def _polarity(y: np.ndarray, kind: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    if kind == "sims":
        # Acc-3 cuts used for CH-SIMS v2.
        out = np.full(y.shape, "neutral", dtype=object)
        out[y <= -0.1] = "negative"
        out[y > 0.1] = "positive"
        return out
    b = np.clip(np.rint(y), -3, 3).astype(np.int32)
    out = np.full(y.shape, "neutral", dtype=object)
    out[b < 0] = "negative"
    out[b > 0] = "positive"
    return out


def _tsne(H: np.ndarray) -> np.ndarray:
    n = H.shape[0]
    perp = float(min(30, max(5, n // 40)))
    emb = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=42,
        early_exaggeration=12.0,
        learning_rate="auto",
        init="pca",
    ).fit_transform(H.astype(np.float64))
    lo = emb.min(axis=0)
    span = np.maximum(emb.max(axis=0) - lo, 1e-9)
    return (emb - lo) / span


def _silhouette(emb: np.ndarray, pol: np.ndarray) -> float:
    labels = np.array([{"negative": 0, "neutral": 1, "positive": 2}[p] for p in pol])
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(emb, labels))


def _style() -> None:
    font_dir = _MY / "loss_curve_analysis" / "fonts"
    for name in ("Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF"):
        path = font_dir / name
        if path.is_file():
            fm.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
    })


def _plot(case: dict, panels: list[tuple[str, np.ndarray, np.ndarray, float]]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.15))
    for ax, (title, emb, pol, sil) in zip(axes, panels):
        for name in ("negative", "neutral", "positive"):
            m = pol == name
            if not np.any(m):
                continue
            ax.scatter(
                emb[m, 0], emb[m, 1],
                s=7 if name == "neutral" else 9,
                c=COLORS[name],
                marker=MARKERS[name],
                linewidths=0.45,
                alpha=0.75,
            )
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(f"{title}\nsilhouette {sil:.3f}", fontsize=9)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    handles = [
        Line2D([0], [0], marker=MARKERS[n], color=COLORS[n], linestyle="None",
               markerfacecolor=COLORS[n] if n == "neutral" else "none",
               markeredgecolor=COLORS[n], markersize=7, label=n)
        for n in ("negative", "neutral", "positive")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(case["title"] + "  test primary embedding", fontsize=12, y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{case['key']}_ablation_tsne.png"
    pdf = OUT / f"{case['key']}_ablation_tsne.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}", flush=True)


def run_case(case: dict) -> None:
    build = _load_build(case["hparams"])
    panels = []
    os.chdir(_MY)
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    for ablation, title in VARIANTS:
        ck_dir = _ckpt_dir(case, ablation)
        ckpt = ck_dir / case["ckpt_name"]
        print(f"=== {case['key']} {ablation} {ckpt} ===", flush=True)
        argv = build(checkpoint_dir=str(ck_dir))
        if ablation != "none":
            argv = argv + ["--ablation", ablation]
        train_mod = _load_train(argv)
        H, y = _collect(train_mod, ckpt, case["kind"])
        pol = _polarity(y, case["kind"])
        emb = _tsne(H)
        sil = _silhouette(emb, pol)
        counts = {name: int(np.sum(pol == name)) for name in ("negative", "neutral", "positive")}
        print(f"  n={len(y)} {counts} silhouette={sil:.3f}", flush=True)
        np.savez(OUT / f"{case['key']}_{ablation}.npz", emb=emb, y=y, pol=pol.astype(str))
        panels.append((title, emb, pol, sil))
    _plot(case, panels)


def main() -> int:
    _style()
    for case in CASES:
        run_case(case)
    print("TSNE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
