#!/usr/bin/env python3
"""MOAC-style single-panel + composite loss curves for the paper Figure.

This is the solidification of the May 2026 one-off Shell heredoc that produced
``results/mosi_trial39_moac_standard/mosi_trial39_moac_{total_loss,task_loss,
token_ib,label_ib,routing}.pdf`` — the files copied into the manuscript
``graph/`` directory.

Data: 10-iteration averaged step losses (``mosi_trial39_stepavg10_losses.csv``).
Style: MOAC Fig. 7-like — Times New Roman, Iterations/Values, small legend, no stage line.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "results" / "mosi_trial39_stepavg10_losses.csv"
DEFAULT_OUT = HERE / "results" / "mosi_trial39_moac_standard"
FONTS_DIR = HERE / "fonts"

# Corefonts Times New Roman (preferred) + renamed Liberation fallbacks.
_TNR_FILES = (
    "Times.TTF",
    "Timesbd.TTF",
    "Timesi.TTF",
    "Timesbi.TTF",
    "TimesNewRoman.ttf",
    "TimesNewRoman-Bold.ttf",
    "TimesNewRoman-Italic.ttf",
    "TimesNewRoman-BoldItalic.ttf",
)

# (csv_column, legend, subcaption, color, output_stem)
SERIES = (
    ("L_total", "Total Loss", "(a) Total Loss", "#77AC77", "mosi_trial39_moac_total_loss"),
    ("L_task", "Task Loss", "(b) Task Loss", "#F28E8B", "mosi_trial39_moac_task_loss"),
    ("L_tib_weighted", "Token IB Loss", "(c) Token IB Loss", "#77AC77", "mosi_trial39_moac_token_ib"),
    ("L_lib_weighted", "Label IB Loss", "(d) Label IB Loss", "#F28E8B", "mosi_trial39_moac_label_ib"),
    ("L_rib_weighted", "Routing Loss", "(e) Routing Loss", "#77AC77", "mosi_trial39_moac_routing"),
)


def _register_times_new_roman() -> None:
    for name in _TNR_FILES:
        path = FONTS_DIR / name
        if path.is_file():
            fm.fontManager.addfont(str(path))


def _apply_rc() -> None:
    _register_times_new_roman()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.45,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_ax(ax: plt.Axes, legend_label: str) -> None:
    ax.set_xlabel("Iterations", labelpad=1.5)
    ax.set_ylabel("Values", labelpad=1.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        framealpha=1.0,
        handlelength=1.8,
        borderpad=0.3,
    )
    ax.tick_params(direction="out", pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return sorted(rows, key=lambda row: float(row["iteration"]))


def _moving_average(values: list[float], window: int) -> list[float]:
    """Centered mean. A short window leaves the raw series as a solid band."""
    n = len(values)
    if window <= 1 or n == 0:
        return values
    half = window // 2
    out = []
    acc = 0.0
    # prefix sums
    prefix = [0.0]
    for v in values:
        acc += v
        prefix.append(acc)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return out


def smooth_rows(rows: list[dict[str, str]], window: int) -> list[dict[str, str]]:
    """Smooth each loss inside a stage so the stage-2 onset of routing stays sharp."""
    if window <= 1:
        return rows
    cols = [spec[0] for spec in SERIES]
    smoothed = [dict(row) for row in rows]
    start = 0
    while start < len(rows):
        stage = rows[start].get("stage", "")
        end = start + 1
        while end < len(rows) and rows[end].get("stage", "") == stage:
            end += 1
        for col in cols:
            raw = [float(rows[i][col]) for i in range(start, end)]
            avg = _moving_average(raw, window)
            for offset, value in enumerate(avg):
                smoothed[start + offset][col] = f"{value:.8g}"
        start = end
    return smoothed


def decimate(rows: list[dict[str, str]], stride: int) -> list[dict[str, str]]:
    """Keep one row every ``stride`` logged bins.

    The CSV is already a 10-iteration mean, but plotting all ~20k bins in a
    two-inch panel paints a solid band. MOAC Figure 4 shows a few thousand
    iterations as a thin polyline; stride 10 of these bins is 100 iterations.
    """
    if stride <= 1:
        return rows
    return rows[::stride]


def plot_composite(rows: list[dict[str, str]], out_dir: Path, ncols: int, prefix: str) -> None:
    x = [float(r["iteration"]) for r in rows]
    n = len(SERIES)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.05 * ncols, 2.0 * nrows))
    flat = list(axes.ravel()) if n > 1 or ncols > 1 else [axes]

    for ax, (col, legend_label, subcaption, color, _stem) in zip(flat, SERIES):
        y = [float(r[col]) for r in rows]
        ax.plot(x, y, color=color, linewidth=1.15, solid_capstyle="round", label=legend_label)
        _style_ax(ax, legend_label)
        ax.text(
            0.5,
            -0.35,
            subcaption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    for ax in flat[n:]:
        ax.axis("off")

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.985,
        bottom=0.10,
        wspace=0.34,
        hspace=0.72,
    )
    stem = f"{prefix}_moac_standard{n}"
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def plot_panels(rows: list[dict[str, str]], out_dir: Path, prefix: str) -> None:
    x = [float(r["iteration"]) for r in rows]
    for col, legend_label, _subcaption, color, stem in SERIES:
        stem = stem.replace("mosi_trial39", prefix)
        y = [float(r[col]) for r in rows]
        fig, ax = plt.subplots(figsize=(4.7, 2.42))
        ax.plot(x, y, color=color, linewidth=1.15, solid_capstyle="round", label=legend_label)
        _style_ax(ax, legend_label)
        fig.subplots_adjust(left=0.16, right=0.985, top=0.98, bottom=0.18)
        pdf = out_dir / f"{stem}.pdf"
        png = out_dir / f"{stem}.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {pdf}")
        print(f"Wrote {png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Step-averaged loss CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="Columns in the composite figure (paper panels are separate; default 2).",
    )
    parser.add_argument(
        "--no-composite",
        action="store_true",
        help="Skip the multi-panel composite PDF/PNG.",
    )
    parser.add_argument(
        "--no-panels",
        action="store_true",
        help="Skip the five single-panel PDF/PNGs used by the manuscript.",
    )
    parser.add_argument(
        "--prefix",
        default="mosi_trial39",
        help="Filename prefix for the five panels and the composite.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Extra centered moving average, in CSV rows. 1 disables it.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Plot one CSV row out of this many. "
             "Each row is a 10-iteration mean, so 10 matches MOAC's point density.",
    )
    args = parser.parse_args()

    rows = decimate(smooth_rows(read_rows(args.csv), max(1, args.smooth)), max(1, args.stride))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _apply_rc()

    if not args.no_composite:
        plot_composite(rows, args.out_dir, ncols=max(1, args.ncols), prefix=args.prefix)
    if not args.no_panels:
        plot_panels(rows, args.out_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
