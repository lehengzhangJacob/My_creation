#!/usr/bin/env python3
"""MOAC-style single-panel + composite loss curves for the paper Figure.

This is the solidification of the May 2026 one-off Shell heredoc that produced
``results/mosi_trial39_moac_standard/mosi_trial39_moac_{total_loss,task_loss,
token_ib,label_ib,routing}.pdf`` — the files copied into the manuscript
``graph/`` directory.

Data: 10-iteration averaged step losses (``mosi_trial39_stepavg10_losses.csv``).
Style: MOAC Fig. 7-like — serif, Iterations/Values, small legend, no stage line.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "results" / "mosi_trial39_stepavg10_losses.csv"
DEFAULT_OUT = HERE / "results" / "mosi_trial39_moac_standard"

# (csv_column, legend, subcaption, color, output_stem)
SERIES = (
    ("L_total", "Total Loss", "(a) Total Loss", "#77AC77", "mosi_trial39_moac_total_loss"),
    ("L_task", "Task Loss", "(b) Task Loss", "#F28E8B", "mosi_trial39_moac_task_loss"),
    ("L_tib_weighted", "Token IB Loss", "(c) Token IB Loss", "#77AC77", "mosi_trial39_moac_token_ib"),
    ("L_lib_weighted", "Label IB Loss", "(d) Label IB Loss", "#F28E8B", "mosi_trial39_moac_label_ib"),
    ("L_rib_weighted", "Routing Loss", "(e) Routing Loss", "#77AC77", "mosi_trial39_moac_routing"),
)


def _apply_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
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
    ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.2)
    ax.tick_params(direction="out", pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return sorted(rows, key=lambda row: float(row["iteration"]))


def plot_composite(rows: list[dict[str, str]], out_dir: Path, ncols: int) -> None:
    x = [float(r["iteration"]) for r in rows]
    n = len(SERIES)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.05 * ncols, 2.0 * nrows))
    flat = list(axes.ravel()) if n > 1 or ncols > 1 else [axes]

    for ax, (col, legend_label, subcaption, color, _stem) in zip(flat, SERIES):
        y = [float(r[col]) for r in rows]
        ax.plot(x, y, color=color, linewidth=0.75, label=legend_label)
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
    stem = f"mosi_trial39_moac_standard{n}"
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def plot_panels(rows: list[dict[str, str]], out_dir: Path) -> None:
    x = [float(r["iteration"]) for r in rows]
    for col, legend_label, _subcaption, color, stem in SERIES:
        y = [float(r[col]) for r in rows]
        fig, ax = plt.subplots(figsize=(3.0, 2.15))
        ax.plot(x, y, color=color, linewidth=0.75, label=legend_label)
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
    args = parser.parse_args()

    rows = read_rows(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _apply_rc()

    if not args.no_composite:
        plot_composite(rows, args.out_dir, ncols=max(1, args.ncols))
    if not args.no_panels:
        plot_panels(rows, args.out_dir)


if __name__ == "__main__":
    main()
