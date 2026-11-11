#!/usr/bin/env python3
"""t-SNE on CH-SIMS v2 test ``h_p`` for PRISM ablations.

This script reads frozen tier3_fresh trial 53 checkpoints under
``ablation_study/runs/`` and writes figures to a separate ``*_tsne`` directory.
It does not train, save checkpoints, or modify existing ablation logs.

Usage (from ``My_creation/``)::

    python ablation_study/tsne_simsv2_prism_ablation.py --max-samples 64 --verbose \
      --output ablation_study/runs/simsv2_tier3_fresh_trial53_tsne/smoke_tsne.png

    python ablation_study/tsne_simsv2_prism_ablation.py --verbose \
      --dump-label-stats ablation_study/runs/simsv2_tier3_fresh_trial53_tsne/label_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

_HERE = Path(__file__).resolve().parent
_MY = _HERE.parent

SIMSV2_CLIP_MIN = -1.0
SIMSV2_CLIP_MAX = 1.0
SIMSV2_BINS_5 = (-1.01, -0.7, -0.1, 0.1, 0.7, 1.01)

ACC5_COLORS = [
    "#2166ac",
    "#67a9cf",
    "#f7f7f7",
    "#ef8a62",
    "#b2182b",
]
ACC5_LABELS = [r"$[-1,-0.7]$", r"$(-0.7,-0.1]$", r"$(-0.1,0.1]$", r"$(0.1,0.7]$", r"$(0.7,1]$"]
ACC5_MARKERS = ("v", "v", "o", "^", "^")

TERNARY_CLASSES: tuple[tuple[str, str, str, str], ...] = (
    ("negative", "negative", "x", "#20B2AA"),
    ("neutral", "neutral", "o", "#0066CC"),
    ("positive", "positive", "+", "#ae017e"),
)

DEFAULT_VARIANTS: tuple[tuple[str, str], ...] = (
    ("none", "PRISM (full)"),
    ("no_ib", "w/o VTB"),
    ("no_mselector", "w/o DPR"),
    ("no_infogate", "w/o InfoGate"),
)


def acc5_bin_index(y: np.ndarray) -> np.ndarray:
    """Match SIMSv2 KuDA Acc-5 label buckets used by ``simsv2_metrics.py``."""
    v = np.clip(np.asarray(y, dtype=np.float64).ravel(), SIMSV2_CLIP_MIN, SIMSV2_CLIP_MAX)
    out = np.full(v.shape, -1, dtype=np.int64)
    for i in range(len(SIMSV2_BINS_5) - 1):
        lo, hi = SIMSV2_BINS_5[i], SIMSV2_BINS_5[i + 1]
        out[np.logical_and(v > lo, v <= hi)] = i
    if np.any(out < 0):
        bad = v[out < 0][:5]
        raise ValueError(f"Some SIMSv2 labels were not bucketed: {bad}")
    return out


def ternary_masks(idx5: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neg = idx5 < 2
    neu = idx5 == 2
    pos = idx5 > 2
    return neg, neu, pos


def label_stats(y: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    if n == 0:
        return {"n": 0}
    qs = np.percentile(y, [5, 25, 50, 75, 95])
    return {
        "n": n,
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "p5": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
        "frac_lt0": float((y < 0).mean()),
    }


def print_label_stats(variant: str, st: dict[str, float | int], counts: np.ndarray) -> None:
    if st.get("n", 0) == 0:
        print(f"[{variant}] no samples")
        return
    print(
        f"[{variant}] n={st['n']} y=[{st['min']:.3f},{st['max']:.3f}] "
        f"mean={st['mean']:.3f} std={st['std']:.3f} "
        f"p5/50/95={st['p5']:.3f}/{st['p50']:.3f}/{st['p95']:.3f} "
        f"P(y<0)={100 * st['frac_lt0']:.1f}%"
    )
    print(f"    Acc-5 bin counts: {counts.tolist()}")


def append_label_stats_csv(path: Path, variant: str, st: dict[str, float | int], counts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "n",
        "min",
        "max",
        "mean",
        "std",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "frac_lt0",
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
    ]
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        row: dict[str, object] = {
            "variant": variant,
            "n": st.get("n", ""),
            "min": st.get("min", ""),
            "max": st.get("max", ""),
            "mean": st.get("mean", ""),
            "std": st.get("std", ""),
            "p5": st.get("p5", ""),
            "p25": st.get("p25", ""),
            "p50": st.get("p50", ""),
            "p75": st.get("p75", ""),
            "p95": st.get("p95", ""),
            "frac_lt0": st.get("frac_lt0", ""),
        }
        for i in range(5):
            row[f"c{i}"] = int(counts[i]) if i < len(counts) else 0
        writer.writerow(row)


def default_ckpt_dir(runs_root: Path, ablation: str) -> Path:
    if ablation == "none":
        return runs_root / "simsv2_tier3_fresh_trial53" / "checkpoints"
    return runs_root / f"simsv2_tier3_fresh_trial53_{ablation}" / "checkpoints"


def configure_train_argv(ablation: str, checkpoint_dir: Path) -> None:
    from simsv2_tier3_fresh_trial53_hparams import build_train_argv

    sys.argv = ["train.py", *build_train_argv(checkpoint_dir=str(checkpoint_dir), ablation=ablation)]


def load_ablation_train_module():
    """Load ablation_study/train.py by absolute path to avoid parent train.py."""
    module_name = "_simsv2_tsne_ablation_train"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, _HERE / "train.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load train module from {_HERE / 'train.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def collect_h_p(train_mod, max_samples: int | None, stage: int) -> tuple[np.ndarray, np.ndarray]:
    train_mod.set_seed(train_mod.args.seed)
    _, _, test_dl, n_opt = train_mod.setup_data()
    model, _, _ = train_mod.build_model(n_opt)
    ckpt_path = os.path.join(train_mod.args.checkpoint_dir, f"infogate_{train_mod.args.dataset}_best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.to(train_mod.DEVICE)
    model.eval()

    hs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_seen = 0
    use_mmsa = train_mod.simsv2_mmsa_enabled()
    with torch.no_grad():
        for batch in test_dl:
            batch = tuple(t.to(train_mod.DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids, input_mask, segment_ids = train_mod.unpack_simsv2_batch(
                batch, use_mmsa
            )
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)
            _, _, _, h_p = train_mod._forward_model(
                model,
                input_ids,
                visual,
                acoustic,
                label_ids,
                stage,
                input_mask,
                segment_ids,
            )
            hs.append(h_p.detach().float().cpu().numpy())
            ys.append(label_ids.detach().cpu().numpy().reshape(-1))
            n_seen += h_p.shape[0]
            if max_samples is not None and n_seen >= max_samples:
                break

    H = np.concatenate(hs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_samples is not None and H.shape[0] > max_samples:
        H = H[:max_samples]
        y = y[:max_samples]

    model.to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return H, y


def unit_box_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    return (x - lo) / span


def run_tsne(
    H: np.ndarray,
    perplexity: float,
    seed: int,
    n_components: int,
    early_exaggeration: float,
    learning_rate: str | float,
    init: str,
) -> np.ndarray:
    n = H.shape[0]
    if n < 4:
        raise ValueError(f"Need at least 4 samples for t-SNE, got {n}")
    perp = min(perplexity, float(n - 1))
    perp = max(perp, 2.0)
    return TSNE(
        n_components=n_components,
        perplexity=perp,
        random_state=seed,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        init=init,
    ).fit_transform(H.astype(np.float64))


def discrete5_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=ACC5_MARKERS[i],
            color="w",
            label=ACC5_LABELS[i],
            markerfacecolor=ACC5_COLORS[i],
            markeredgecolor="#2a2a2a",
            markersize=6,
            linewidth=0,
        )
        for i in range(5)
    ]


def ternary_legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for _key, label, marker, color in TERNARY_CLASSES:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color=color,
                label=label,
                markerfacecolor=color if marker == "o" else "none",
                markeredgecolor=color,
                markersize=6,
                linewidth=0,
            )
        )
    return handles


def parse_learning_rate(value: str) -> str | float:
    raw = value.strip().lower()
    if raw == "auto":
        return "auto"
    lr = float(raw)
    if lr <= 0:
        raise ValueError("--learning-rate must be positive")
    return lr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=_MY / "ablation_study" / "runs")
    ap.add_argument(
        "--output",
        type=Path,
        default=_MY
        / "ablation_study"
        / "runs"
        / "simsv2_tier3_fresh_trial53_tsne"
        / "ablation_tsne_prism_simsv2.png",
        help="Output figure path; writes both .png and .pdf",
    )
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--early-exaggeration", type=float, default=24.0)
    ap.add_argument("--learning-rate", type=str, default="auto")
    ap.add_argument("--init", type=str, choices=("pca", "random"), default="pca")
    ap.add_argument("--n-components", type=int, choices=(2, 3), default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=2)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--variants", type=str, default="none,no_ib,no_mselector,no_infogate")
    ap.add_argument(
        "--color-mode",
        choices=("discrete5", "ternary", "continuous"),
        default="ternary",
        help="ternary: neg/neu/pos (recommended for paper figures). discrete5: Acc-5 buckets.",
    )
    ap.add_argument("--marker-size", type=float, default=1.35)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dump-label-stats", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=480)
    ap.add_argument("--fig-scale", type=float, default=1.25)
    args = ap.parse_args()

    try:
        learning_rate = parse_learning_rate(args.learning_rate)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    runs_root = args.runs_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    pairs: list[tuple[str, str]] = []
    for token in args.variants.split(","):
        ablation = token.strip()
        if not ablation:
            continue
        title = next((t for a, t in DEFAULT_VARIANTS if a == ablation), ablation)
        pairs.append((ablation, title))
    if not pairs:
        print("error: --variants resolved to an empty list", file=sys.stderr)
        return 2

    os.chdir(_MY)
    # Force imports such as ``import train`` to resolve to ablation_study/train.py,
    # not the parent My_creation/train.py that lacks PRISM ablation switches.
    sys.path = [p for p in sys.path if p != str(_HERE)]
    sys.path.insert(0, str(_HERE))

    n_panels = len(pairs)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(math.ceil(n_panels / ncols))
    fig_scale = float(args.fig_scale)
    fig_w = (3.4 * ncols + 0.8) * fig_scale
    fig_h = (3.0 * nrows + 1.2) * fig_scale
    n_components = int(args.n_components)

    if n_components == 3:
        fig = plt.figure(figsize=(fig_w, fig_h * 1.12))
        axes_flat = [fig.add_subplot(nrows, ncols, i + 1, projection="3d") for i in range(nrows * ncols)]
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=False)
        axes_flat = [axes] if n_panels == 1 else np.atleast_1d(axes).ravel().tolist()

    legend_handles = discrete5_legend_handles() if args.color_mode == "discrete5" else ternary_legend_handles()
    area = float(args.marker_size) * fig_scale * fig_scale

    for panel_i, (ax, (ablation, title)) in enumerate(zip(axes_flat, pairs)):
        ckpt_dir = default_ckpt_dir(runs_root, ablation)
        configure_train_argv(ablation, ckpt_dir)
        train_mod = load_ablation_train_module()

        H, y = collect_h_p(train_mod, args.max_samples, args.stage)
        idx = acc5_bin_index(y)
        counts = np.bincount(idx, minlength=5)
        st = label_stats(y)
        if args.verbose:
            print_label_stats(ablation, st, counts)
        if args.dump_label_stats is not None:
            append_label_stats_csv(args.dump_label_stats.expanduser().resolve(), ablation, st, counts)

        emb = run_tsne(
            H,
            float(args.perplexity),
            int(args.seed),
            n_components,
            float(args.early_exaggeration),
            learning_rate,
            str(args.init),
        )

        plot_emb = unit_box_normalize(emb) if args.color_mode == "ternary" else emb
        panel_tag = f"({chr(ord('a') + panel_i)}) "

        if args.color_mode == "continuous":
            ax.set_title(f"{panel_tag}{title}", fontsize=10)
            if n_components == 3:
                sc = ax.scatter(plot_emb[:, 0], plot_emb[:, 1], plot_emb[:, 2], c=y, cmap="RdBu_r", s=area, alpha=0.55)
            else:
                sc = ax.scatter(plot_emb[:, 0], plot_emb[:, 1], c=y, cmap="RdBu_r", s=area, alpha=0.55)
            fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
        elif args.color_mode == "ternary":
            neg_m, neu_m, pos_m = ternary_masks(idx)
            n_neg, n_neu, n_pos = int(neg_m.sum()), int(neu_m.sum()), int(pos_m.sum())
            ternary_s = max(0.35 * area, min(7.5, float(args.marker_size) * fig_scale * fig_scale * 0.52))
            for key, _label, marker, color in TERNARY_CLASSES:
                if key == "negative":
                    sel, lw = neg_m, 0.55
                elif key == "neutral":
                    sel, lw = neu_m, 0.0
                else:
                    sel, lw = pos_m, 0.5
                if not np.any(sel):
                    continue
                kw = dict(c=[color], marker=marker, s=ternary_s, alpha=0.52)
                if marker == "o":
                    kw.update(linewidths=0, edgecolors="none")
                else:
                    kw["linewidths"] = lw
                if n_components == 3:
                    ax.scatter(plot_emb[sel, 0], plot_emb[sel, 1], plot_emb[sel, 2], **kw)
                else:
                    ax.scatter(plot_emb[sel, 0], plot_emb[sel, 1], **kw)
            ax.set_title(
                f"{panel_tag}{title}\nneg {n_neg} | neu {n_neu} | pos {n_pos}",
                fontsize=10,
                ma="left",
            )
            ticks = np.arange(0.0, 1.01, 0.25)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            if n_components == 3:
                ax.set_zlim(0.0, 1.0)
                ax.set_zticks(ticks)
        else:
            ax.set_title(f"{panel_tag}{title}\nAcc-5 bins: {counts.tolist()}", fontsize=10)
            for i in range(5):
                sel = idx == i
                if not np.any(sel):
                    continue
                if n_components == 3:
                    ax.scatter(
                        emb[sel, 0],
                        emb[sel, 1],
                        emb[sel, 2],
                        c=[ACC5_COLORS[i]],
                        marker=ACC5_MARKERS[i],
                        s=area,
                        alpha=0.55,
                        linewidths=0.15,
                        edgecolors="#2a2a2a",
                    )
                else:
                    ax.scatter(
                        emb[sel, 0],
                        emb[sel, 1],
                        c=[ACC5_COLORS[i]],
                        marker=ACC5_MARKERS[i],
                        s=area,
                        alpha=0.55,
                        linewidths=0.15,
                        edgecolors="#2a2a2a",
                    )

        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        if n_components == 3:
            ax.set_zlabel("t-SNE 3", fontsize=9)
            ax.view_init(elev=22, azim=-58)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    if args.color_mode in ("discrete5", "ternary"):
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.012),
        )
        bottom = 0.12
    else:
        bottom = 0.06

    fig.suptitle("CH-SIMS v2 PRISM ablation t-SNE (test h_p)", fontsize=12)
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
