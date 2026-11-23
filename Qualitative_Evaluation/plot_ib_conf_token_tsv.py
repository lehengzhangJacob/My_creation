#!/usr/bin/env python3
"""Plot token-level VTB confidences from ``dump_ib_conf_token_level.py`` TSV output.

Usage (from ``My_creation/``)::

    conda run -n ITHP5090 python Qualitative_Evaluation/plot_ib_conf_token_tsv.py \\
      --input Qualitative_Evaluation/results/mosei_trial70_idx2856_ib_conf_tokens.tsv \\
      --output Qualitative_Evaluation/results/mosei_trial70_idx2856_ib_conf_tokens.png

Heatmap (modalities × token positions)::

    conda run -n ITHP5090 python Qualitative_Evaluation/plot_ib_conf_token_tsv.py \\
      --input .../mosei_trial70_idx2856_ib_conf_tokens.tsv \\
      --output .../mosei_trial70_idx2856_ib_conf_hm.png --kind heatmap

Narrow color scale when values sit in a band (e.g.\ 0.5--0.7)::

    ... --kind heatmap --vscale data

Full [0,1] scale (default, backward-compatible)::

    ... --kind heatmap --vscale full

Tall layout + diverging colormap (purple band near 0.5, not white)::

    ... --kind heatmap --layout tall --norm twoslope --vscale data
    # default cmap becomes ``ib_purple_div``; or e.g.\ ``--cmap twilight``
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any

import numpy as np

IB_CONF_SHORT_LABEL = {
    "ib_conf_t": "text",
    "ib_conf_a": "acoustic",
    "ib_conf_v": "visual",
    "ib_conf_h": "HCF",
    "ib_conf_fused": "fused",
}

Z_BASE_LINE = {"ib_conf_fused": 1, "ib_conf_a": 2, "ib_conf_v": 3, "ib_conf_h": 3, "ib_conf_t": 4}
LINE_COLORS = {
    "ib_conf_t": "#2ca02c",
    "ib_conf_a": "#1f77b4",
    "ib_conf_v": "#ff7f0e",
    "ib_conf_h": "#8c564b",
    "ib_conf_fused": "#d62728",
}


def load_ib_conf_token_rows(path: str, include_pad: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if include_pad or int(row.get("valid", "0")) == 1:
                rows.append(row)
    return rows


def resolve_ib_conf_plot_columns(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    if not rows:
        return [], []
    conf_cols = [c for c in rows[0].keys() if c.startswith("ib_conf_")]
    preferred = ["ib_conf_t", "ib_conf_a", "ib_conf_v", "ib_conf_h", "ib_conf_fused"]
    plot_cols = [c for c in preferred if c in conf_cols]
    for c in sorted(conf_cols):
        if c not in plot_cols:
            plot_cols.append(c)

    def _max_abs_diff(c1: str, c2: str) -> float:
        return max(abs(float(r[c1]) - float(r[c2])) for r in rows)

    notes: list[str] = []
    tol = 1e-5
    if "ib_conf_fused" in plot_cols:
        for c in ("ib_conf_a", "ib_conf_t", "ib_conf_v", "ib_conf_h"):
            if c in plot_cols and _max_abs_diff("ib_conf_fused", c) < tol:
                plot_cols = [x for x in plot_cols if x != "ib_conf_fused"]
                notes.append(f"ib_conf_fused ≡ {c} (fused not drawn)")
                break
    return plot_cols, notes


def ib_conf_heatmap_matrix(rows: list[dict[str, str]], plot_cols: list[str]) -> list[list[float]]:
    return [[float(row[c]) for row in rows] for c in plot_cols]


def heatmap_vmin_vmax(
    mat: list[list[float]],
    *,
    vscale: str,
    vmin_user: float | None,
    vmax_user: float | None,
) -> tuple[float, float]:
    """Return (vmin, vmax) for ``imshow``, clamped to [0, 1]."""
    if vmin_user is not None and vmax_user is not None:
        return max(0.0, min(1.0, float(vmin_user))), max(0.0, min(1.0, float(vmax_user)))
    if vmin_user is not None or vmax_user is not None:
        flat = [x for row in mat for x in row]
        lo, hi = min(flat), max(flat)
        if vmin_user is not None:
            lo = max(0.0, min(1.0, float(vmin_user)))
        if vmax_user is not None:
            hi = max(0.0, min(1.0, float(vmax_user)))
        if lo >= hi:
            hi = min(1.0, lo + 1e-3)
        return lo, hi
    if vscale == "full":
        return 0.0, 1.0
    flat = [x for row in mat for x in row]
    lo, hi = min(flat), max(flat)
    if lo == hi:
        lo = max(0.0, lo - 0.02)
        hi = min(1.0, hi + 0.02)
    span = hi - lo
    pad = max(span * 0.02, 1e-4)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def ib_purple_center_cmap() -> Any:
    """Diverging-style map for ``[0,1]`` conf: blue (low) → purple (mid) → red (high)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "ib_purple_div",
        [
            "#053061",
            "#2166ac",
            "#4393c3",
            "#7b3294",
            "#ae017e",
            "#d73027",
            "#a50026",
        ],
        N=256,
    )


def resolve_heatmap_cmap(cmap_arg: str | Any) -> Any:
    if isinstance(cmap_arg, str) and cmap_arg == "ib_purple_div":
        return ib_purple_center_cmap()
    return cmap_arg


def _transpose_modality_token(mat: list[list[float]]) -> np.ndarray:
    """``mat`` rows = modalities, cols = tokens -> array shape (n_tok, n_ch)."""
    return np.asarray(mat, dtype=np.float64).T


def _twoslope_norm(vmin: float, vmax: float, vcenter: float = 0.5) -> Any:
    from matplotlib.colors import TwoSlopeNorm

    lo, hi = float(vmin), float(vmax)
    vc = float(vcenter)
    if not (lo < vc < hi):
        if hi <= vc:
            hi = min(1.0, vc + 1e-3)
        if lo >= vc:
            lo = max(0.0, vc - 1e-3)
        if not (lo < vc < hi):
            lo, hi = max(0.0, vc - 0.2), min(1.0, vc + 0.2)
    return TwoSlopeNorm(vmin=lo, vcenter=vc, vmax=hi)


def draw_ib_conf_heatmap(
    fig: Any,
    ax: Any,
    rows: list[dict[str, str]],
    plot_cols: list[str],
    notes: list[str],
    *,
    title: str,
    cmap: str = "viridis",
    vscale: str = "full",
    vmin: float | None = None,
    vmax: float | None = None,
    layout: str = "wide",
    norm_style: str = "linear",
) -> Any:
    """Draw heatmap on ``ax``; colorbar on ``fig``. ``layout``: ``wide`` (x=tokens) or ``tall`` (y=tokens)."""
    mat = ib_conf_heatmap_matrix(rows, plot_cols)
    n_tok = len(rows)
    n_ch = len(plot_cols)
    vmin_v, vmax_v = heatmap_vmin_vmax(mat, vscale=vscale, vmin_user=vmin, vmax_user=vmax)

    if layout == "tall":
        arr = _transpose_modality_token(mat)
    else:
        arr = np.asarray(mat, dtype=np.float64)

    if norm_style == "twoslope":
        norm = _twoslope_norm(vmin_v, vmax_v, 0.5)
        cmap_resolved = resolve_heatmap_cmap(cmap)
        im = ax.imshow(arr, aspect="auto", norm=norm, cmap=cmap_resolved, interpolation="nearest")
        cbar_label = r"VTB conf.\ $\sigma(-\log\sigma^2)$ (mean$_D$); purple$\approx$0.5"
    else:
        cmap_resolved = resolve_heatmap_cmap(cmap)
        im = ax.imshow(
            arr,
            aspect="auto",
            vmin=vmin_v,
            vmax=vmax_v,
            cmap=cmap_resolved,
            interpolation="nearest",
        )
        cbar_label = "sigmoid(-logvar) (mean over D)"

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)

    tok_labels = [row.get("token", str(i)).replace("\n", " ")[:28] for i, row in enumerate(rows)]

    if layout == "tall":
        ax.set_xlabel("VTB slot")
        ax.set_ylabel("Token (English subword; top to bottom = left-to-right sequence)")
        ax.set_xticks(range(n_ch))
        ax.set_xticklabels([IB_CONF_SHORT_LABEL.get(c, c) for c in plot_cols], fontsize=9, rotation=12, ha="right")
        tick_step = max(1, n_tok // 55)
        tick_ix = list(range(0, n_tok, tick_step))
        if n_tok > 1 and tick_ix[-1] != n_tok - 1:
            tick_ix.append(n_tok - 1)
        ax.set_yticks(tick_ix)
        ax.set_yticklabels([tok_labels[i] for i in tick_ix], fontsize=5.5)
        ax.tick_params(axis="y", length=1, pad=1)
    else:
        ax.set_yticks(range(n_ch))
        ax.set_yticklabels([IB_CONF_SHORT_LABEL.get(c, c) for c in plot_cols], fontsize=9)
        ax.set_xlabel("English subword (tokenizer order; valid positions)")
        tick_step = max(1, n_tok // 25)
        tick_ix = list(range(0, n_tok, tick_step))
        if n_tok > 1 and tick_ix[-1] != n_tok - 1:
            tick_ix.append(n_tok - 1)
        ax.set_xticks(tick_ix)
        ax.set_xticklabels([tok_labels[i] for i in tick_ix], rotation=65, ha="right", fontsize=7)

    ax.set_title(title)
    if notes:
        uniq = sorted(set(notes))
        ax.text(
            0.99,
            1.02,
            " · ".join(uniq),
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
        )
    return im


def save_heatmap_figures(
    out_paths: list[str],
    rows: list[dict[str, str]],
    plot_cols: list[str],
    notes: list[str],
    *,
    title: str,
    cmap: str,
    vscale: str,
    vmin: float | None,
    vmax: float | None,
    layout: str = "wide",
    norm_style: str = "linear",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from pathlib import Path

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
            fm.fontManager.addfont(str(_path))
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n_tok = len(rows)
    n_ch = len(plot_cols)
    if layout == "tall":
        fig_w = max(5.0, min(9.0, 0.9 * n_ch + 3.8))
        fig_h = min(56.0, max(5.0, 0.11 * n_tok + 2.2))
    else:
        fig_w = max(6.0, min(28.0, 0.18 * n_tok + 2.5))
        fig_h = max(2.2, 0.45 * n_ch + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    draw_ib_conf_heatmap(
        fig,
        ax,
        rows,
        plot_cols,
        notes,
        title=title,
        cmap=cmap,
        vscale=vscale,
        vmin=vmin,
        vmax=vmax,
        layout=layout,
        norm_style=norm_style,
    )
    fig.tight_layout()
    for out in out_paths:
        outp = os.path.abspath(out)
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        fig.savefig(outp, bbox_inches="tight")
        print(f"Wrote {outp}")
    plt.close(fig)


def save_heatmap_figure(
    fig_path: str,
    rows: list[dict[str, str]],
    plot_cols: list[str],
    notes: list[str],
    *,
    title: str,
    cmap: str,
    vscale: str,
    vmin: float | None,
    vmax: float | None,
    layout: str = "wide",
    norm_style: str = "linear",
) -> None:
    save_heatmap_figures(
        [fig_path],
        rows,
        plot_cols,
        notes,
        title=title,
        cmap=cmap,
        vscale=vscale,
        vmin=vmin,
        vmax=vmax,
        layout=layout,
        norm_style=norm_style,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=str, required=True, help="Tab-separated token conf TSV")
    ap.add_argument("--output", type=str, required=True, help="Output PNG path")
    ap.add_argument(
        "--kind",
        type=str,
        choices=("line", "heatmap", "both"),
        default="line",
        help="line: curves over positions; heatmap: modalities×tokens; both: *_line.png + *_heatmap.png",
    )
    ap.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap: matplotlib name, or ``ib_purple_div`` (blue→purple→red, mid purple). "
        "Built-ins like ``twilight`` also have a non-white center band.",
    )
    ap.add_argument(
        "--vscale",
        type=str,
        choices=("full", "data"),
        default="full",
        help="Heatmap color limits: full=[0,1]; data=min/max of plotted matrix (+margin, clamped)",
    )
    ap.add_argument("--vmin", type=float, default=None, help="Heatmap vmin (clamped to [0,1]; use with --vmax)")
    ap.add_argument("--vmax", type=float, default=None, help="Heatmap vmax (clamped to [0,1]; use with --vmin)")
    ap.add_argument(
        "--layout",
        type=str,
        choices=("wide", "tall"),
        default="wide",
        help="wide: x=tokens (many columns); tall: y=tokens, x=VTB slots (better for long sequences)",
    )
    ap.add_argument(
        "--norm",
        type=str,
        choices=("linear", "twoslope"),
        default="linear",
        dest="norm_style",
        help="linear: vmin/vmax; twoslope: TwoSlopeNorm centered at 0.5 (pair with ``ib_purple_div`` or ``RdBu_r``)",
    )
    ap.add_argument(
        "--include-pad",
        action="store_true",
        help="Also plot padded positions (default: only valid=1 rows)",
    )
    args = ap.parse_args()

    if args.norm_style == "twoslope" and args.cmap == "viridis":
        args.cmap = "ib_purple_div"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib required", file=sys.stderr)
        return 1

    if not os.path.isfile(args.input):
        print(f"ERROR: not found: {args.input}", file=sys.stderr)
        return 1

    rows = load_ib_conf_token_rows(args.input, args.include_pad)
    if not rows:
        print("ERROR: no rows to plot", file=sys.stderr)
        return 1

    plot_cols, notes = resolve_ib_conf_plot_columns(rows)
    if not plot_cols:
        print("ERROR: no ib_conf_* columns", file=sys.stderr)
        return 1

    x = list(range(len(rows)))
    labels = [row.get("token", str(i)).replace("\n", " ")[:12] for i, row in enumerate(rows)]

    def _plot_line(fig_path: str) -> None:
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.22), 4.2), dpi=150)

        draw_order = sorted(plot_cols, key=lambda c: (Z_BASE_LINE.get(c, 2), c))
        for col in draw_order:
            ys = [float(row[col]) for row in rows]
            z = Z_BASE_LINE.get(col, 2) * 10
            ax.plot(
                x,
                ys,
                marker=".",
                linewidth=1.6,
                label=col,
                color=LINE_COLORS.get(col),
                zorder=z,
            )

        ax.set_xlabel("token index (valid positions in file order)")
        ax.set_ylabel("mean_D sigmoid(-logvar)")
        ax.set_title(os.path.basename(args.input))
        ystack = [float(row[c]) for row in rows for c in plot_cols]
        lo, hi = min(ystack), max(ystack)
        pad = max(0.02, (hi - lo) * 0.15)
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.legend(loc="upper right", fontsize=8)
        if notes:
            uniq = sorted(set(notes))
            ax.text(
                0.01,
                0.02,
                "\n".join(uniq),
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
            )
        ax.grid(True, alpha=0.3)

        tick_step = max(1, len(labels) // 25)
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels([labels[i] for i in x[::tick_step]], rotation=65, ha="right", fontsize=7)

        fig.tight_layout()
        out = os.path.abspath(fig_path)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")

    base, ext = os.path.splitext(args.output)
    if ext == "":
        ext = ".png"
    if args.kind == "line":
        _plot_line(args.output)
    elif args.kind == "heatmap":
        save_heatmap_figure(
            args.output,
            rows,
            plot_cols,
            notes,
            title=os.path.basename(args.input),
            cmap=args.cmap,
            vscale=args.vscale,
            vmin=args.vmin,
            vmax=args.vmax,
            layout=args.layout,
            norm_style=args.norm_style,
        )
    else:
        _plot_line(f"{base}_line{ext}")
        save_heatmap_figure(
            f"{base}_heatmap{ext}",
            rows,
            plot_cols,
            notes,
            title=os.path.basename(args.input),
            cmap=args.cmap,
            vscale=args.vscale,
            vmin=args.vmin,
            vmax=args.vmax,
            layout=args.layout,
            norm_style=args.norm_style,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
