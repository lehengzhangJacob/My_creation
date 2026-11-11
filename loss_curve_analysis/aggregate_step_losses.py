#!/usr/bin/env python3
"""Average per-step loss CSV rows over fixed iteration windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "runs/mosi_trial39_step_loss/step_losses.csv"
DEFAULT_OUTPUT = HERE / "results/mosi_trial39_stepavg10_losses.csv"

MEAN_COLUMNS = [
    "L_total",
    "L_total_scaled",
    "L_task",
    "ib_logged",
    "L_tib_raw",
    "L_lib_raw",
    "L_rib_raw",
    "L_rib_kl_raw",
    "L_rib_balance_raw",
    "L_tib_weighted",
    "L_lib_weighted",
    "L_rib_weighted",
    "alpha_ib",
    "selector_rib_weight",
    "gradient_accumulation_step",
]

FIELDNAMES = [
    "dataset",
    "bucket",
    "window",
    "n_steps",
    "iteration_start",
    "iteration_end",
    "iteration",
    "epoch_start",
    "epoch_end",
    "stage",
    *MEAN_COLUMNS,
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return sorted(rows, key=lambda row: int(row["global_step"]))


def aggregate(rows: list[dict[str, str]], window: int) -> list[dict[str, float | int | str]]:
    if window <= 0:
        raise ValueError("--window must be positive")
    out: list[dict[str, float | int | str]] = []
    for bucket_idx, start in enumerate(range(0, len(rows), window), start=1):
        chunk = rows[start : start + window]
        steps = [int(row["global_step"]) for row in chunk]
        epochs = [int(row["epoch"]) for row in chunk]
        stages = [int(row["stage"]) for row in chunk]
        row_out: dict[str, float | int | str] = {
            "dataset": chunk[0]["dataset"],
            "bucket": bucket_idx,
            "window": window,
            "n_steps": len(chunk),
            "iteration_start": min(steps),
            "iteration_end": max(steps),
            "iteration": sum(steps) / len(steps),
            "epoch_start": min(epochs),
            "epoch_end": max(epochs),
            "stage": max(stages),
        }
        for column in MEAN_COLUMNS:
            row_out[column] = sum(float(row[column]) for row in chunk) / len(chunk)
        out.append(row_out)
    return out


def write_rows(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    rows = read_rows(args.input)
    aggregated = aggregate(rows, args.window)
    write_rows(aggregated, args.output)
    print(
        f"Wrote {args.output} from {len(rows)} step rows "
        f"({len(aggregated)} buckets, window={args.window})"
    )


if __name__ == "__main__":
    main()
