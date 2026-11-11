#!/usr/bin/env python3
"""Parse per-epoch DPR diagnostics already printed in training logs.

Each epoch block looks like::

    Epoch 12/50  [stage 2]
      Diag  w=[a:0.21 l:0.58 v:0.21]  target=[...]  primary=[a:0.20 l:0.59 v:0.21]

No retraining is required. Outputs one CSV per dataset and a primary-share plot.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MY = HERE.parent

EPOCH_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)/\d+\s+\[stage\s+(?P<stage>\d+)\]"
)
TRIPLE_RE = re.compile(
    r"(?P<name>w|target|primary)=\["
    r"a:(?P<a>[-+0-9.eE]+)\s+l:(?P<l>[-+0-9.eE]+)\s+v:(?P<v>[-+0-9.eE]+)\]"
)

DEFAULT_LOGS = (
    ("mosei", MY / "ablation_study/runs/mosei_phase1_trial70/train.log", "full"),
    ("simsv2", MY / "ablation_study/runs/simsv2_tier3_fresh_trial53/train.log", "full"),
    ("mosi", MY / "ablation_study/runs/mosi_more_trial39/train.log", "full"),
)


def parse_routing_diag(text: str, dataset: str, run: str) -> list[dict[str, str]]:
    """Return one row per Diag line, attached to the latest Epoch header."""
    rows: list[dict[str, str]] = []
    epoch = ""
    stage = ""
    for line in text.splitlines():
        header = EPOCH_RE.search(line)
        if header:
            epoch = header.group("epoch")
            stage = header.group("stage")
            continue
        if "Diag" not in line:
            continue
        found = {match.group("name"): match for match in TRIPLE_RE.finditer(line)}
        if set(found) != {"w", "target", "primary"}:
            continue
        row = {"dataset": dataset, "run": run, "epoch": epoch, "stage": stage}
        for name, match in found.items():
            for modality in ("a", "l", "v"):
                row[f"{name}_{modality}"] = match.group(modality)
        rows.append(row)
    return rows


def _stage2_epoch(rows: list[dict[str, str]]) -> float | None:
    for row in rows:
        if row.get("stage") == "2" and row.get("epoch"):
            return float(row["epoch"])
    return None


def plot_primary_share(rows: list[dict[str, str]], output: Path, title: str) -> None:
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows if row["epoch"]]
    if not epochs:
        return
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for key, label, color in (
        ("primary_a", "acoustic", "#4C78A8"),
        ("primary_l", "language", "#F28E8B"),
        ("primary_v", "visual", "#77AC77"),
    ):
        ax.plot(
            epochs,
            [float(row[key]) for row in rows],
            label=label,
            color=color,
            linewidth=1.4,
        )
    stage2 = _stage2_epoch(rows)
    if stage2 is not None:
        ax.axvline(stage2, color="0.45", linestyle="--", linewidth=0.9)
        ax.text(stage2, 0.98, "stage 2", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=8, color="0.35")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hard-primary share")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        print(f"Wrote empty {path}")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} epochs)")


def _discover_router_logs() -> list[tuple[str, Path, str]]:
    found = []
    for path in sorted((HERE / "logs").glob("*router*_s128.log")):
        name = path.name
        dataset = name.split("_", 1)[0]
        run = path.stem
        found.append((dataset, path, run))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    specs = list(DEFAULT_LOGS) + _discover_router_logs()
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for dataset, path, run in specs:
        if not path.is_file():
            print(f"Missing {path}")
            continue
        rows = parse_routing_diag(path.read_text(encoding="utf-8", errors="replace"), dataset, run)
        by_dataset.setdefault(dataset, []).extend(rows)
        print(f"{path.name}: {len(rows)} diag rows")
    for dataset, rows in by_dataset.items():
        write_csv(rows, args.out_dir / f"routing_dynamics_{dataset}.csv")
        full = [row for row in rows if row["run"] == "full"]
        plot_primary_share(
            full or rows,
            args.out_dir / f"routing_dynamics_{dataset}.pdf",
            f"{dataset.upper()} hard-primary share",
        )


if __name__ == "__main__":
    main()
