#!/usr/bin/env python3
"""Summarize hard and soft DPR routing from per-sample dump CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np


def _mean_std(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def summarize(path: str) -> dict[str, object]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"routing CSV contains no rows: {path}")

    required = {
        "primary_idx", "primary_name", "w_a", "w_l", "w_v",
        "w_max", "route_entropy", "abs_err",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    n = len(rows)
    names = [row["primary_name"] for row in rows]
    counts = Counter(names)
    weights = {
        "acoustic": np.asarray([float(row["w_a"]) for row in rows]),
        "language": np.asarray([float(row["w_l"]) for row in rows]),
        "visual": np.asarray([float(row["w_v"]) for row in rows]),
    }
    entropy = np.asarray([float(row["route_entropy"]) for row in rows])
    maximum = np.asarray([float(row["w_max"]) for row in rows])
    errors = np.asarray([float(row["abs_err"]) for row in rows])
    indices = np.asarray(
        [int(row.get("split_index", index)) for index, row in enumerate(rows)]
    )
    primary_indices = np.asarray([int(row["primary_idx"]) for row in rows])

    quantile_edges = np.quantile(errors, [0.0, 0.25, 0.5, 0.75, 1.0])
    error_groups: list[dict[str, object]] = []
    for quartile in range(4):
        lower = quantile_edges[quartile]
        upper = quantile_edges[quartile + 1]
        if quartile == 3:
            mask = (errors >= lower) & (errors <= upper)
        else:
            mask = (errors >= lower) & (errors < upper)
        group: dict[str, object] = {
            "quartile": quartile + 1,
            "lower_abs_error": float(lower),
            "upper_abs_error": float(upper),
            "n": int(mask.sum()),
            "mean_abs_error": float(errors[mask].mean()),
            "mean_route_entropy": float(entropy[mask].mean()),
            "mean_w_max": float(maximum[mask].mean()),
        }
        for modality, values in weights.items():
            group[f"mean_w_{modality}"] = float(values[mask].mean())
        error_groups.append(group)

    return {
        "source": os.path.abspath(path),
        "n": n,
        "hard_primary": {
            modality: {
                "count": int(counts.get(modality, 0)),
                "percentage": 100.0 * float(counts.get(modality, 0)) / n,
            }
            for modality in ("acoustic", "language", "visual")
        },
        "soft_weights": {
            modality: _mean_std(values)
            for modality, values in weights.items()
        },
        "route_entropy": _mean_std(entropy),
        "w_max": _mean_std(maximum),
        "mean_abs_error": float(errors.mean()),
        "non_language": {
            "count": int((primary_indices != 1).sum()),
            "percentage": 100.0 * float((primary_indices != 1).sum()) / n,
            "split_indices": indices[primary_indices != 1].tolist(),
        },
        "error_quartiles": error_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="DATASET=CSV",
        help="May be repeated.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload: dict[str, object] = {}
    for item in args.input:
        if "=" not in item:
            raise SystemExit(f"invalid --input {item!r}; expected DATASET=CSV")
        dataset, path = item.split("=", 1)
        dataset = dataset.strip()
        if not dataset:
            raise SystemExit("dataset name cannot be empty")
        if dataset in payload:
            raise SystemExit(f"duplicate dataset: {dataset}")
        payload[dataset] = summarize(os.path.abspath(path))

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"Wrote {output}")
    for dataset, summary in payload.items():
        hard = summary["hard_primary"]
        print(
            f"{dataset}: n={summary['n']} "
            f"A={hard['acoustic']['percentage']:.4f}% "
            f"L={hard['language']['percentage']:.4f}% "
            f"V={hard['visual']['percentage']:.4f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
