#!/usr/bin/env python3
"""Plot weighted MOSI/MOSEI loss curves from parsed CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

DEFAULT_INPUTS = (
    RESULTS_DIR / "mosi_trial39_losses.csv",
    RESULTS_DIR / "mosei_trial70_losses.csv",
)

SERIES = (
    ("L_total", "Total"),
    ("L_task", "Task"),
    ("L_tib_weighted", "alpha_IB * L_tib"),
    ("L_lib_weighted", "alpha_IB * L_lib"),
    ("L_rib_weighted", "lambda_rib * L_rib"),
)

MAIN_SERIES = (
    ("L_total", r"$\mathcal{L}$"),
    ("L_task", r"$\mathcal{L}_{task}$"),
)

REG_SERIES = (
    ("L_tib_weighted", r"$\alpha_{IB}\mathcal{L}_{tib}$"),
    ("L_lib_weighted", r"$\alpha_{IB}\mathcal{L}_{lib}$"),
    ("L_rib_weighted", r"$\lambda_{rib}\mathcal{L}_{rib}$"),
)

INDIVIDUAL_SERIES = (
    ("L_total", r"Total loss $\mathcal{L}$"),
    ("L_task", r"Task loss $\mathcal{L}_{task}$"),
    ("L_tib_weighted", r"Weighted token IB $\alpha_{IB}\mathcal{L}_{tib}$"),
    ("L_lib_weighted", r"Weighted label IB $\alpha_{IB}\mathcal{L}_{lib}$"),
    ("L_rib_weighted", r"Weighted routing $\lambda_{rib}\mathcal{L}_{rib}$"),
)

ROUTING_COLUMN = "L_rib_weighted"


def _component_series(hide_routing: bool) -> tuple[tuple[str, str], ...]:
    if not hide_routing:
        return INDIVIDUAL_SERIES
    return tuple(series for series in INDIVIDUAL_SERIES if series[0] != ROUTING_COLUMN)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run parse_train_log_losses.py first.")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    if "iteration" in rows[0]:
        return sorted(rows, key=lambda row: float(row["iteration"]))
    if "global_step" in rows[0]:
        return sorted(rows, key=lambda row: int(row["global_step"]))
    return sorted(rows, key=lambda row: int(row["epoch"]))


def plot_case(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    dataset = rows[0]["dataset"].upper()
    trial = rows[0]["trial"].replace("_", " ")
    epochs = [int(row["epoch"]) for row in rows]
    stage1_epochs = int(rows[0]["stage1_epochs"])

    for column, label in SERIES:
        values = [float(row[column]) for row in rows]
        linewidth = 2.0 if column == "L_total" else 1.4
        ax.plot(epochs, values, linewidth=linewidth, label=label)

    stage2_start = stage1_epochs + 1
    ax.axvline(stage2_start, color="0.35", linestyle="--", linewidth=1.0)
    ax.text(
        stage2_start,
        ax.get_ylim()[1] * 0.96,
        f"stage 2 starts @ {stage2_start}",
        ha="left",
        va="top",
        fontsize=9,
        color="0.25",
    )

    ax.set_title(f"{dataset} ({trial})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25, linewidth=0.7)


def _epochs(rows: list[dict[str, str]]) -> list[int]:
    return [int(row["epoch"]) for row in rows]


def _values(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def _add_stage_marker(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    stage2_start = int(rows[0]["stage1_epochs"]) + 1
    ax.axvline(stage2_start, color="0.45", linestyle="--", linewidth=0.9)
    ax.text(
        stage2_start,
        0.96,
        f"stage 2 @ {stage2_start}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        color="0.3",
    )


def _plot_series(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    series: tuple[tuple[str, str], ...],
) -> None:
    epochs = _epochs(rows)
    for column, label in series:
        ax.plot(
            epochs,
            _values(rows, column),
            marker="o",
            markersize=2.4,
            linewidth=1.45,
            label=label,
        )
    _add_stage_marker(ax, rows)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Epoch")


def plot_paper(all_rows: list[list[dict[str, str]]], output: Path, also_png: bool) -> None:
    if len(all_rows) != 2:
        raise ValueError("--paper expects MOSI and MOSEI CSVs, in that order.")

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.6),
        constrained_layout=True,
    )

    for row_idx, rows in enumerate(all_rows):
        dataset = rows[0]["dataset"].upper()
        trial = rows[0]["trial"].replace("_", " ")

        main_ax = axes[row_idx][0]
        reg_ax = axes[row_idx][1]

        _plot_series(main_ax, rows, MAIN_SERIES)
        _plot_series(reg_ax, rows, REG_SERIES)

        main_ax.set_title(f"{dataset}: total and task")
        reg_ax.set_title(f"{dataset}: weighted regularizers")
        main_ax.set_ylabel("Epoch-averaged loss")
        reg_ax.set_ylabel("Epoch-averaged loss")

        if row_idx == 0:
            main_ax.legend(loc="upper right", frameon=False)
            reg_ax.legend(loc="upper right", frameon=False)

        # Keep the canonical trial visible without overcrowding the title.
        main_ax.text(
            0.02,
            0.04,
            trial,
            transform=main_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")

    if also_png:
        png_path = output.with_suffix(".png")
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        print(f"Wrote {png_path}")


def _add_axis_padding(ax: plt.Axes, values: list[float]) -> None:
    low, high = min(values), max(values)
    if low == high:
        pad = 0.05 if high == 0 else abs(high) * 0.1
    else:
        pad = (high - low) * 0.12
    ax.set_ylim(low - pad, high + pad)


def _individual_output_path(output: Path, dataset: str) -> Path:
    return output.with_name(f"{output.stem}_{dataset.lower()}{output.suffix}")


def plot_individual(
    all_rows: list[list[dict[str, str]]],
    output: Path,
    also_png: bool,
    hide_routing: bool = False,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    series_to_plot = _component_series(hide_routing)
    ncols = 2
    nrows = (len(series_to_plot) + ncols - 1) // ncols
    for rows in all_rows:
        dataset = rows[0]["dataset"].upper()
        trial = rows[0]["trial"].replace("_", " ")
        epochs = _epochs(rows)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7.2, 2.1 * nrows),
            constrained_layout=True,
        )
        flat_axes = list(axes.ravel())

        for ax, (column, title) in zip(flat_axes, series_to_plot):
            values = _values(rows, column)
            ax.plot(
                epochs,
                values,
                marker="o",
                markersize=2.2,
                linewidth=0.85,
                alpha=0.9,
            )
            _add_stage_marker(ax, rows)
            _add_axis_padding(ax, values)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Epoch-avg value")
            ax.grid(True, alpha=0.22, linewidth=0.55)

        for ax in flat_axes[len(series_to_plot) :]:
            ax.axis("off")
        fig.suptitle(f"{dataset} loss components ({trial})", fontsize=12)

        pdf_path = _individual_output_path(output, rows[0]["dataset"])
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Wrote {pdf_path}")

        if also_png:
            png_path = pdf_path.with_suffix(".png")
            fig.savefig(png_path, dpi=220, bbox_inches="tight")
            print(f"Wrote {png_path}")
        plt.close(fig)


def _step_output_path(output: Path, dataset: str, multi: bool) -> Path:
    if not multi:
        return output
    return output.with_name(f"{output.stem}_{dataset.lower()}{output.suffix}")


def _add_step_stage_marker(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    stage2_rows = [row for row in rows if int(float(row["stage"])) >= 2]
    if not stage2_rows:
        return
    stage2_start = float(stage2_rows[0].get("iteration_start", stage2_rows[0]["iteration"]))
    ax.axvline(stage2_start, color="0.45", linestyle="--", linewidth=0.9)
    ax.text(
        stage2_start,
        0.96,
        f"stage 2 @ {stage2_start:g}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        color="0.3",
    )


def plot_moac_step(
    all_rows: list[list[dict[str, str]]],
    output: Path,
    also_png: bool,
    hide_routing: bool = False,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    multi = len(all_rows) > 1
    series_to_plot = _component_series(hide_routing)
    ncols = 2
    nrows = (len(series_to_plot) + ncols - 1) // ncols
    for rows in all_rows:
        dataset = rows[0]["dataset"].upper()
        iterations = [float(row["iteration"]) for row in rows]
        window = rows[0].get("window", "10")

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7.2, 2.1 * nrows),
            constrained_layout=True,
        )
        flat_axes = list(axes.ravel())

        for ax, (column, title) in zip(flat_axes, series_to_plot):
            values = _values(rows, column)
            ax.plot(
                iterations,
                values,
                linewidth=0.75,
                alpha=0.9,
            )
            _add_step_stage_marker(ax, rows)
            _add_axis_padding(ax, values)
            ax.set_title(title)
            ax.set_xlabel("Iterations")
            ax.set_ylabel(f"{window}-iter avg")
            ax.grid(True, alpha=0.22, linewidth=0.55)

        for ax in flat_axes[len(series_to_plot) :]:
            ax.axis("off")
        fig.suptitle(f"{dataset} MOAC-style loss curves", fontsize=12)

        pdf_path = _step_output_path(output, rows[0]["dataset"], multi)
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Wrote {pdf_path}")

        if also_png:
            png_path = pdf_path.with_suffix(".png")
            fig.savefig(png_path, dpi=220, bbox_inches="tight")
            print(f"Wrote {png_path}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_INPUTS),
        help="Parsed CSV files to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "mosi_mosei_loss_curves.pdf",
        help="Output figure path.",
    )
    parser.add_argument(
        "--also-png",
        action="store_true",
        help="Also save a PNG next to the requested output.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use a 2x2 paper-ready layout with separate regularizer scale.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Write one figure per dataset with each loss in its own subplot.",
    )
    parser.add_argument(
        "--moac-step",
        action="store_true",
        help="Plot every-N-iteration averaged step-loss CSVs in MOAC style.",
    )
    parser.add_argument(
        "--hide-routing",
        action="store_true",
        help="Omit the weighted routing regularizer from individual/MOAC component plots.",
    )
    args = parser.parse_args()

    all_rows = [read_rows(path) for path in args.inputs]
    modes = [args.paper, args.individual, args.moac_step]
    if sum(bool(mode) for mode in modes) > 1:
        raise ValueError("Choose only one of --paper, --individual, or --moac-step.")
    if args.paper:
        plot_paper(all_rows, args.output, args.also_png)
        return
    if args.individual:
        plot_individual(all_rows, args.output, args.also_png, args.hide_routing)
        return
    if args.moac_step:
        plot_moac_step(all_rows, args.output, args.also_png, args.hide_routing)
        return

    fig, axes = plt.subplots(
        len(all_rows),
        1,
        figsize=(8.0, 3.8 * len(all_rows)),
        sharex=False,
        constrained_layout=True,
    )
    if len(all_rows) == 1:
        axes = [axes]

    for ax, rows in zip(axes, all_rows):
        plot_case(ax, rows)

    axes[0].legend(loc="upper right", ncol=2, frameon=False, fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")

    if args.also_png:
        png_path = args.output.with_suffix(".png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
