#!/usr/bin/env python3
"""Paper figure: token-level confidence isolation (retrain + post-hoc)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

_QE = Path(__file__).resolve().parent
_ROOT = _QE.parent.parent
_OUT = _ROOT / "overleaf_69e83a58" / "graph" / "conf_isolation_appendix.pdf"
_INTERVENTION = _QE / "results" / "reviewer_conf_intervention_effects.json"

# Short ticks under narrow facets; full names in the shared legend.
TICKS = ["Tok", "Mod", "Unif", "None"]
LEGEND_LABELS = ["Token", "Modality", r"Uniform $1/L$", "No conf."]
RETRAIN = {
    "CMU-MOSEI": [0.499, 0.500, 0.527, 0.508],
    "CH-SIMS v2": [0.291, 0.301, 0.311, 0.301],
}
COLORS = ["#1b6b4a", "#4c8c6a", "#c45c26", "#8a8a8a"]


def _load_intervention() -> dict:
    with open(_INTERVENTION, encoding="utf-8") as f:
        return json.load(f)


def _style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    inter = _load_intervention()
    fig = plt.figure(figsize=(8.4, 3.35))
    gs = GridSpec(
        1, 2, figure=fig, width_ratios=[1.2, 1.0], wspace=0.30,
        left=0.07, right=0.99, bottom=0.18, top=0.82,
    )

    # --- Panel (a): two faceted retrain plots ---
    gs_a = gs[0].subgridspec(1, 2, wspace=0.40)
    axes_a = [fig.add_subplot(gs_a[0, 0]), fig.add_subplot(gs_a[0, 1])]
    x = np.arange(len(TICKS))
    for ax, (dataset, vals) in zip(axes_a, RETRAIN.items()):
        bars = ax.bar(
            x, vals, width=0.62, color=COLORS, edgecolor="white", linewidth=0.4, zorder=2,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.008,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=False,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(TICKS, fontsize=7.5)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_ylim(0.24, 0.58)
        ax.set_title(dataset, fontsize=9, pad=3)
        _style_ax(ax)
    axes_a[0].set_ylabel(r"MAE $\downarrow$")
    fig.text(0.32, 0.94, "(a) Retrained confidence modes", ha="center", fontsize=10)
    fig.legend(
        handles=[Patch(facecolor=c, edgecolor="white", linewidth=0.4) for c in COLORS],
        labels=LEGEND_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.32, 0.02),
        ncol=4,
        frameon=False,
        fontsize=7,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    # --- Panel (b): post-hoc ---
    ax = fig.add_subplot(gs[0, 1])
    modes = ["uniform", "inverse", "modality"]
    mode_labels = [r"Unif. $1/L$", "Inverse", "Mod. mean"]
    ds_keys = [("mosei", "CMU-MOSEI"), ("simsv2", "CH-SIMS v2")]
    x = np.arange(len(modes), dtype=float) * 1.2
    width = 0.32
    handles_b = []
    for j, (key, _label) in enumerate(ds_keys):
        comps = inter[key]["comparisons"]
        deltas = [comps[m]["delta_mae_intervention_minus_original"] for m in modes]
        lowers = [comps[m]["ci95_lower"] for m in modes]
        uppers = [comps[m]["ci95_upper"] for m in modes]
        yerr = np.vstack(
            [np.array(deltas) - np.array(lowers), np.array(uppers) - np.array(deltas)]
        )
        xpos = x + (j - 0.5) * width
        bars = ax.bar(
            xpos,
            deltas,
            width=width * 0.90,
            color=["#2f5d50", "#7aa899"][j],
            edgecolor="white",
            linewidth=0.4,
            yerr=yerr,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
            zorder=2,
        )
        handles_b.append(bars[0])
        for bar, v, hi in zip(bars, deltas, uppers):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                hi + 0.004,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                clip_on=False,
            )
    ax.axhline(0.0, color="#666666", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylabel(r"$\Delta$MAE (interv. $-$ original)")
    ax.set_title("(b) Post-hoc interventions (weights fixed)", fontsize=10, pad=8)
    ax.set_ylim(-0.005, 0.118)
    _style_ax(ax)
    ax.legend(
        handles_b,
        ["CMU-MOSEI", "CH-SIMS v2"],
        loc="upper right",
        frameon=False,
        fontsize=7,
        handlelength=1.15,
        handletextpad=0.35,
        borderaxespad=0.2,
    )

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, bbox_inches="tight")
    print(f"wrote {_OUT}")


if __name__ == "__main__":
    # Match Elsevier CAS body text (STIX serif used by cas-dc).
    from matplotlib import font_manager

    _stix = Path("/usr/share/fonts/opentype/stix/STIXGeneral-Regular.otf")
    if _stix.is_file():
        font_manager.fontManager.addfont(str(_stix))
        for _extra in (
            "/usr/share/fonts/opentype/stix/STIXGeneral-Italic.otf",
            "/usr/share/fonts/opentype/stix/STIXGeneral-Bold.otf",
            "/usr/share/fonts/opentype/stix/STIXGeneral-BoldItalic.otf",
        ):
            if Path(_extra).is_file():
                font_manager.fontManager.addfont(_extra)
        _serif_name = "STIXGeneral"
    else:
        _serif_name = "Liberation Serif"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [_serif_name, "TeXGyreTermes", "Nimbus Roman", "Liberation Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    main()
