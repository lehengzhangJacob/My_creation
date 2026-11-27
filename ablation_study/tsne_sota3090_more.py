#!/usr/bin/env python3
"""t-SNE for the ablation families trained by runs/ablation_rest_3090/queue.py.

One family per process: ``--family mustard_trial1239``.
Each panel is an independent t-SNE of test-set ``h_p``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def _load_families():
    """Load the family table without putting that directory on sys.path.

    That directory contains ``queue.py``, which would shadow the stdlib ``queue`` module.
    """
    spec = importlib.util.spec_from_file_location(
        "ablation_rest_families",
        _MY / "runs" / "ablation_rest_3090" / "families.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FAM = _load_families()
ABLATIONS = _FAM.ABLATIONS
ablation_dir = _FAM.ablation_dir
ckpt_for = _FAM.ckpt_for
family_by_key = _FAM.family_by_key

OUT = _HERE / "runs" / "sota3090_tsne"
VARIANTS = (("none", "PRISM (full)"),) + tuple(
    (name, {"no_ib": "w/o VTB", "no_mselector": "w/o DPR", "no_infogate": "w/o InfoGate"}[name])
    for name in ABLATIONS
)
TERNARY = ("negative", "neutral", "positive")
T_COLORS = {"negative": "#20B2AA", "neutral": "#0066CC", "positive": "#ae017e"}
T_MARKERS = {"negative": "x", "neutral": "o", "positive": "+"}
B_COLORS = ("#20B2AA", "#ae017e")
B_MARKERS = ("x", "+")


def _argv_for(fam: dict, ablation: str) -> list[str]:
    if fam["trial"]:
        fe = str(_MY / "fixed_experiment")
        # Do not leave this directory at sys.path[0]: it ships an older
        # infogate_modules.py and would be imported instead of ablation_study's.
        inserted = fe not in sys.path
        if inserted:
            sys.path.insert(0, fe)
        from sota_2026_11_trials import build_argv
        if inserted:
            sys.path.remove(fe)
        argv = build_argv(fam["trial"], str(ablation_dir(fam, ablation) / "checkpoints"))
    else:
        cmd = json.loads(Path(fam["cmd_json"]).read_text())["cmd"]
        argv = list(cmd[3:])
        for i, tok in enumerate(argv):
            if tok == "--checkpoint_dir":
                argv[i + 1] = str(ablation_dir(fam, ablation) / "checkpoints")
    if ablation != "none" and "--ablation" not in argv:
        argv += ["--ablation", ablation]
    return argv


def _pin_ablation_modules() -> None:
    """Force InfoGate imports to come from ablation_study, not fixed_experiment."""
    abl = str(_HERE)
    if abl in sys.path:
        sys.path.remove(abl)
    sys.path.insert(0, abl)
    for name in (
        "infogate_modules",
        "infogate_modules_fixed",
        "albert_infogate",
        "deberta_infogate",
        "bert_infogate",
    ):
        path = _HERE / f"{name}.py"
        cur = sys.modules.get(name)
        cur_file = Path(getattr(cur, "__file__", "") or "")
        if cur is not None and cur_file.resolve() == path.resolve():
            continue
        sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)


def _load_module(filename: str, argv: list[str], mod_name: str):
    sys.argv = [filename, *argv]
    _pin_ablation_modules()
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(mod_name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _collect(mod, ckpt: Path, loader: str) -> tuple[np.ndarray, np.ndarray]:
    mod.set_seed(int(mod.args.seed))
    _, _, test_dl, n_opt = mod.setup_data()
    model, _, _ = mod.build_model(n_opt)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.to(mod.DEVICE)
    model.eval()
    hs, ys = [], []
    use_hcf = loader == "classify" and mod.effective_hcf_dim > 0
    with torch.no_grad():
        for batch in test_dl:
            batch = tuple(t.to(mod.DEVICE) for t in batch)
            if loader == "classify":
                input_ids, visual, acoustic, hcf, label_ids = mod._unpack_batch(batch, use_hcf)
                _, _, _, h_p = model(input_ids, visual, acoustic, hcf=hcf, stage=2)
            else:
                input_ids, visual, acoustic, label_ids = batch
                visual = visual.squeeze(1)
                acoustic = acoustic.squeeze(1)
                _, _, _, h_p = model(input_ids, visual, acoustic, labels=label_ids, stage=2)
            hs.append(h_p.detach().float().cpu().numpy())
            ys.append(label_ids.detach().cpu().numpy().reshape(-1))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(hs, 0), np.concatenate(ys, 0)


def _labels(y: np.ndarray, kind: str) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if kind == "binary":
        return (y.astype(np.float64) >= 0.5).astype(np.int32)
    b = np.clip(np.rint(y.astype(np.float64)), -3, 3).astype(np.int32)
    out = np.full(b.shape, "neutral", dtype=object)
    out[b < 0] = "negative"
    out[b > 0] = "positive"
    return out


def _tsne(H: np.ndarray) -> np.ndarray:
    n = H.shape[0]
    perp = float(min(30, max(5, (n - 1) / 3.0 if n < 90 else n // 40)))
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


def _silhouette(emb: np.ndarray, labels: np.ndarray) -> float:
    if labels.dtype == object:
        ids = np.array([{"negative": 0, "neutral": 1, "positive": 2}[p] for p in labels])
    else:
        ids = labels.astype(np.int32)
    if len(set(ids.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(emb, ids))


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


def _plot(fam: dict, panels: list[tuple[str, np.ndarray, np.ndarray, float]]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.15))
    binary = fam["kind"] == "binary"
    names = fam["classes"] if binary else TERNARY
    for ax, (title, emb, labels, sil) in zip(axes, panels):
        if binary:
            for i, name in enumerate(names):
                m = labels == i
                if not np.any(m):
                    continue
                ax.scatter(
                    emb[m, 0], emb[m, 1], s=16, c=B_COLORS[i], marker=B_MARKERS[i],
                    linewidths=0.6, alpha=0.85,
                )
        else:
            for name in TERNARY:
                m = labels == name
                if not np.any(m):
                    continue
                ax.scatter(
                    emb[m, 0], emb[m, 1],
                    s=7 if name == "neutral" else 9,
                    c=T_COLORS[name], marker=T_MARKERS[name],
                    linewidths=0.45, alpha=0.75,
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
    if binary:
        handles = [
            Line2D([0], [0], marker=B_MARKERS[i], color=B_COLORS[i], linestyle="None",
                   markeredgecolor=B_COLORS[i], markersize=7, label=names[i])
            for i in range(2)
        ]
    else:
        handles = [
            Line2D([0], [0], marker=T_MARKERS[n], color=T_COLORS[n], linestyle="None",
                   markerfacecolor=T_COLORS[n] if n == "neutral" else "none",
                   markeredgecolor=T_COLORS[n], markersize=7, label=n)
            for n in TERNARY
        ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(fam["title"] + "  test primary embedding", fontsize=12, y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{fam['key']}_ablation_tsne.png"
    pdf = OUT / f"{fam['key']}_ablation_tsne.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}", flush=True)


def run_family(key: str) -> None:
    fam = family_by_key(key)
    os.chdir(_MY)
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    panels = []
    filename = "train_classify.py" if fam["loader"] == "classify" else "train.py"
    for ablation, title in VARIANTS:
        ckpt = ckpt_for(fam, ablation)
        print(f"=== {key} {ablation} {ckpt} ===", flush=True)
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        mod = _load_module(filename, _argv_for(fam, ablation), "_tsne_more_train")
        H, y = _collect(mod, ckpt, fam["loader"])
        labels = _labels(y, fam["kind"])
        emb = _tsne(H)
        sil = _silhouette(emb, labels)
        if fam["kind"] == "binary":
            counts = {fam["classes"][i]: int(np.sum(labels == i)) for i in (0, 1)}
        else:
            counts = {name: int(np.sum(labels == name)) for name in TERNARY}
        print(f"  n={len(y)} {counts} silhouette={sil:.3f}", flush=True)
        np.savez(
            OUT / f"{key}_{ablation}.npz",
            emb=emb, y=np.asarray(y).reshape(-1),
            labels=np.asarray(labels).astype(str),
        )
        panels.append((title, emb, labels, sil))
    full_sil = panels[0][3]
    rest = ", ".join(f"{title} {sil:.3f}" for title, _, _, sil in panels[1:])
    print(f"COMPARE full {full_sil:.3f} vs {rest}", flush=True)
    _plot(fam, panels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    args = ap.parse_args()
    _style()
    run_family(args.family)
    print("TSNE_FAMILY_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
