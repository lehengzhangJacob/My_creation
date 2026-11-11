#!/usr/bin/env python3
"""EMOE-style sensitivity figure for CMU-MOSEI (trial 70 sweeps).

Reads ``summary.csv`` (default: ``sensitivity_analysis/results/mosei_trial70/summary.csv``).
Produce it with::

    python sensitivity_analysis/run_mosei_trial70_sensitivity.py aggregate \\
        --runs-root runs/sensitivity_mosei/trial70 \\
        --summary-out sensitivity_analysis/results/mosei_trial70/summary.csv

Visual layout follows EMOE (CVPR 2025) Fig.4: one row of panels, each with
dual y-axes — Acc-7 (%, left, orange + circles) and MAE (right, light blue +
diamonds), with the MAE axis inverted (lower MAE toward the top).

Default y-ranges: Acc-7 **35–75%%**; MAE **0.4–0.6** (inverted: lower error at top).
Override with ``--acc7-ylim`` / ``--mae-ylim``, or ``auto`` for data-driven limits.

Example::

    cd My_creation
    python3 sensitivity_analysis/run_mosei_trial70_sensitivity.py aggregate \\
        --runs-root runs/sensitivity_mosei/trial70 \\
        --summary-out sensitivity_analysis/results/mosei_trial70/summary.csv
    python3 sensitivity_analysis/plot_mosei_sensitivity_emoe_style.py \\
        --output sensitivity_analysis/results/mosei_trial70/mosei_sensitivity_emoe_style.pdf

By default **``α_IB``** uses a **log x-axis** with ``$10^{-3}$`` … ``$10^{-1}$``-style
tick labels (matches CMU-MOSEI trial-70 reference). Other panels are **linear** unless
you pass ``--xlog-axes`` (comma-separated names, e.g. ``beta_ib``).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MultipleLocator, NullFormatter

_SENS = Path(__file__).resolve().parent

# Default y-limits (requested display band).
DEFAULT_ACC7_YLIM: tuple[float, float] = (35.0, 75.0)
DEFAULT_MAE_YLIM: tuple[float, float] = (0.4, 0.6)
_DEFAULT_RESULTS = _SENS / "results" / "mosei_trial70"
_DEFAULT_SUMMARY = _DEFAULT_RESULTS / "summary.csv"
_DEFAULT_OUTPUT = _DEFAULT_RESULTS / "mosei_trial70_sensitivity_emoe_style.pdf"

DEFAULT_AXES_ORDER = (
    "beta_ib",
    "mse_weight",
    "alpha_ib",
    "selector_target_temp",
    "selector_rib_weight",
)

# X-axis labels (below each panel): prefer Greek letters for hyper-parameters.
AXIS_XLABEL = {
    "beta_ib": r"$\beta_{\mathrm{IB}}$",
    "mse_weight": r"$\rho_{\mathrm{mse}}$",
    "alpha_ib": r"$\alpha_{\mathrm{IB}}$",
    "selector_target_temp": r"$\tau_{\mathrm{sel}}$",
    "selector_rib_weight": r"$\gamma_{\mathrm{RIB}}$",
}


def _parse_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(summary_csv: Path) -> list[dict[str, str]]:
    if not summary_csv.is_file():
        print(f"ERROR: summary CSV not found: {summary_csv}", file=sys.stderr)
        sys.exit(1)
    with summary_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_for_axis(rows: list[dict[str, str]], axis: str) -> list[tuple[float, float, float]]:
    """Return list of (x_value, acc7_frac, mae) sorted by x."""
    out: list[tuple[float, float, float]] = []
    for row in rows:
        if (row.get("axis") or "").strip() != axis:
            continue
        xv = _parse_float(str(row.get("value", "")))
        acc7 = _parse_float(str(row.get("Acc-7", "")))
        mae = _parse_float(str(row.get("MAE", "")))
        if xv is None or acc7 is None or mae is None:
            print(f"WARN: skip row (axis={axis!r}): {row}", file=sys.stderr)
            continue
        if not (math.isfinite(xv) and math.isfinite(acc7) and math.isfinite(mae)):
            continue
        out.append((xv, acc7, mae))
    out.sort(key=lambda t: t[0])
    return out


def _acc7_ylim(series_pct: list[float]) -> tuple[float, float]:
    if not series_pct:
        return DEFAULT_ACC7_YLIM
    lo, hi = min(series_pct), max(series_pct)
    pad = max(2.0, (hi - lo) * 0.15)
    return lo - pad, hi + pad


def _mae_ylim(
    series: list[float],
    lo_hi: tuple[float, float] | None,
    pad: float = 0.02,
) -> tuple[float, float]:
    """Return (high, low) for set_ylim so MAE increases downward (EMOE style)."""
    if lo_hi is not None:
        lo, hi = lo_hi
        return hi, lo
    if not series:
        return DEFAULT_MAE_YLIM[1], DEFAULT_MAE_YLIM[0]
    lo, hi = min(series), max(series)
    return hi + pad, lo - pad


def plot(
    *,
    summary_csv: Path,
    output_path: Path,
    axes_order: tuple[str, ...],
    xlog_axes: set[str],
    acc7_ylim: tuple[float, float] | None,
    mae_ylim: tuple[float, float] | None,
    strict: bool,
) -> None:
    rows = load_rows(summary_csv)
    plt.rcParams.update({"font.size": 13})
    fig, axs = plt.subplots(
        1,
        len(axes_order),
        figsize=(4.65 * len(axes_order), 4.95),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.55, wspace=0.05, hspace=0.0)


    if len(axes_order) == 1:
        axs = [axs]

    any_data = False
    for ax, axis in zip(axs, axes_order):
        data = rows_for_axis(rows, axis)
        if not data:
            msg = f"no data for axis={axis}"
            if strict:
                print(f"ERROR: {msg}", file=sys.stderr)
                sys.exit(1)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(
                AXIS_XLABEL.get(axis, axis), fontsize=15, labelpad=10,
            )
            continue

        any_data = True
        xs = [d[0] for d in data]
        acc7_pct = [d[1] * 100.0 for d in data]
        maes = [d[2] for d in data]

        color_acc = "#E07020"
        color_mae = "#7EB6D9"

        (line_acc,) = ax.plot(
            xs,
            acc7_pct,
            color=color_acc,
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=7,
            label=r"$ACC_7$",
        )
        ax.set_ylabel(r"$ACC_7$", color=color_acc, fontsize=14)
        ax.tick_params(axis="y", labelsize=13, labelcolor=color_acc)
        if acc7_ylim is not None:
            ax.set_ylim(acc7_ylim[0], acc7_ylim[1])
            span_a = acc7_ylim[1] - acc7_ylim[0]
        else:
            y_lo, y_hi = _acc7_ylim(acc7_pct)
            ax.set_ylim(y_lo, y_hi)
            span_a = y_hi - y_lo
        acc_step = 5 if span_a <= 45 else 10
        ax.yaxis.set_major_locator(MultipleLocator(acc_step))
        # α_IB: fixed log x (reference figure); other axes: log only if listed in xlog_axes.
        use_log_x = axis == "alpha_ib" or axis in xlog_axes
        if use_log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.xaxis.set_major_formatter(LogFormatterMathtext())
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelsize=13, pad=2)

        ax2 = ax.twinx()
        (line_mae,) = ax2.plot(
            xs,
            maes,
            color=color_mae,
            marker="D",
            linestyle="-",
            linewidth=1.8,
            markersize=6,
            label="MAE",
        )
        ax2.set_ylabel("MAE", color=color_mae, fontsize=14)
        ax2.tick_params(axis="y", labelsize=13, labelcolor=color_mae)
        m_hi, m_lo = _mae_ylim(maes, mae_ylim)
        ax2.set_ylim(m_hi, m_lo)
        if mae_ylim is not None:
            span_m = float(mae_ylim[1]) - float(mae_ylim[0])
        else:
            span_m = (max(maes) - min(maes)) if maes else 0.05
        mae_step = 0.05 if span_m >= 0.15 else (0.02 if span_m >= 0.06 else 0.01)
        ax2.yaxis.set_major_locator(MultipleLocator(mae_step))

        ax.grid(True, axis="y", alpha=0.35)
        ax.set_xlabel(
            AXIS_XLABEL.get(axis, axis), fontsize=15, labelpad=11,
        )

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(
            h1 + h2, l1 + l2, loc="upper right", fontsize=12, framealpha=0.9,
        )

    if not any_data:
        print("ERROR: no plottable data in any panel", file=sys.stderr)
        sys.exit(1)

    for ax, axis in zip(axs, axes_order):
        use_log_x = axis == "alpha_ib" or axis in xlog_axes
        if not use_log_x:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(16)
                lbl.set_ha("right")
            continue
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(18)
            lbl.set_ha("right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = output_path.expanduser().resolve()
    suffix = out.suffix.lower()
    if suffix == ".pdf":
        fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.28)
    elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg"):
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.28)
    else:
        fig.savefig(out, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--summary-csv",
        type=Path,
        default=_DEFAULT_SUMMARY,
        help=f"Path to summary.csv (default: {_DEFAULT_SUMMARY})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output figure path (.pdf / .png / …; default: {_DEFAULT_OUTPUT})",
    )
    ap.add_argument(
        "--axes",
        type=str,
        default=",".join(DEFAULT_AXES_ORDER),
        help="Comma-separated axis names (default: standard trial70 axes)",
    )
    ap.add_argument(
        "--xlog-axes",
        type=str,
        default="",
        help=(
            "Comma-separated axes for optional log-scale x (default: none). "
            "The α_IB panel is always log-x; this flag only affects other names "
            "(e.g. beta_ib). Log axes use $10^{k}$-style tick labels."
        ),
    )
    ap.add_argument(
        "--acc7-ylim",
        type=str,
        default="35,75",
        help='Left y-axis limits "lo,hi" in Acc-7 %%. Default 35,75. Use "auto" for data-driven.',
    )
    ap.add_argument(
        "--mae-ylim",
        type=str,
        default="0.4,0.6",
        help='MAE axis band "lo,hi" (inverted: good MAE at top). Default 0.4,0.6. Use "auto" for data-driven.',
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any requested axis has no rows",
    )
    args = ap.parse_args()

    axes_order = tuple(a.strip() for a in args.axes.split(",") if a.strip())
    if not axes_order:
        print("ERROR: --axes is empty", file=sys.stderr)
        return 2

    xlog = {a.strip() for a in args.xlog_axes.split(",") if a.strip()}

    acc7_ylim: tuple[float, float] | None
    s_acc = (args.acc7_ylim or "").strip().lower()
    if s_acc in ("", "auto"):
        acc7_ylim = None if s_acc == "auto" else DEFAULT_ACC7_YLIM
    else:
        parts = [p.strip() for p in args.acc7_ylim.split(",")]
        if len(parts) != 2:
            print('ERROR: --acc7-ylim must be "lo,hi" or auto', file=sys.stderr)
            return 2
        acc7_ylim = (float(parts[0]), float(parts[1]))

    mae_ylim: tuple[float, float] | None
    s_mae = (args.mae_ylim or "").strip().lower()
    if s_mae in ("", "auto"):
        mae_ylim = None if s_mae == "auto" else DEFAULT_MAE_YLIM
    else:
        parts = [p.strip() for p in args.mae_ylim.split(",")]
        if len(parts) != 2:
            print('ERROR: --mae-ylim must be "lo,hi" or auto', file=sys.stderr)
            return 2
        lo, hi = float(parts[0]), float(parts[1])
        mae_ylim = (lo, hi)

    plot(
        summary_csv=args.summary_csv.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        axes_order=axes_order,
        xlog_axes=xlog,
        acc7_ylim=acc7_ylim,
        mae_ylim=mae_ylim,
        strict=args.strict,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())