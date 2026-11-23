#!/usr/bin/env python3
"""Paper figure: token-level confidence isolation (retrained modes only).

Panel (b) post-hoc (weights fixed) was dropped — deltas were too small to read
as a useful claim. Layout is a single-row two-dataset MAE bar chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_QE = Path(__file__).resolve().parent
_ROOT = _QE.parent.parent
_OUTS = [
    _ROOT / "overleaf_InfoFusion" / "graph" / "conf_isolation_appendix.pdf",
    _ROOT / "Information_Fusion_manuscript" / "graph" / "conf_isolation_appendix.pdf",
]

TICKS = ["Tok", "Mod", "Unif", "None"]
LEGEND_LABELS = ["Token", "Modality", r"Uniform $1/L$", "No conf."]
RETRAIN = {
    "CMU-MOSEI": [0.499, 0.500, 0.527, 0.508],
    "CH-SIMS v2": [0.291, 0.301, 0.311, 0.301],
}
# Tight bands around each dataset so bar heights (and gaps) are readable.
YLIM = {
    "CMU-MOSEI": (0.485, 0.540),
    "CH-SIMS v2": (0.280, 0.325),
}
COLORS = ["#1b6b4a", "#4c8c6a", "#c45c26", "#8a8a8a"]


def _style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.55, color="#d0d0d0", zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(5.6, 2.35),
        gridspec_kw={"wspace": 0.55},
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.28, top=0.82)

    x = np.arange(len(TICKS))
    for ax, (dataset, vals) in zip(axes, RETRAIN.items()):
        lo, hi = YLIM[dataset]
        span = hi - lo
        bars = ax.bar(
            x, vals, width=0.68, color=COLORS, edgecolor="white", linewidth=0.35, zorder=2,
        )
        for bar, v in zip(bars, vals):
            y_txt = min(v + 0.06 * span, hi - 0.08 * span)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_txt,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                clip_on=True,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(TICKS, fontsize=7)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_ylim(lo, hi)
        ax.set_title(dataset, fontsize=8.5, pad=6)
        _style_ax(ax)
    axes[0].set_ylabel(r"MAE $\downarrow$", fontsize=8)
    fig.suptitle("Retrained confidence modes", fontsize=9.5, y=0.98)
    fig.legend(
        handles=[Patch(facecolor=c, edgecolor="white", linewidth=0.35) for c in COLORS],
        labels=LEGEND_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.12),
        ncol=4,
        frameon=False,
        fontsize=6.5,
        handlelength=0.95,
        handletextpad=0.30,
        columnspacing=0.75,
    )

    for out in _OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    from matplotlib import font_manager

    _tnr_dir = Path(__file__).resolve().parents[1] / "loss_curve_analysis" / "fonts"
    for _name in (
        "Times.TTF",
        "Timesbd.TTF",
        "Timesi.TTF",
        "Timesbi.TTF",
        "TimesNewRoman.ttf",
        "TimesNewRoman-Bold.ttf",
        "TimesNewRoman-Italic.ttf",
        "TimesNewRoman-BoldItalic.ttf",
    ):
        _path = _tnr_dir / _name
        if _path.is_file():
            font_manager.fontManager.addfont(str(_path))

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    main()
