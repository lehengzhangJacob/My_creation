#!/usr/bin/env python3
"""Paired MAE analysis for full-split confidence interventions."""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


def _paired_mae_bootstrap(
    original_error: np.ndarray,
    intervention_error: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, float | int]:
    """Estimate intervention-minus-original MAE with a paired bootstrap."""
    if original_error.shape != intervention_error.shape:
        raise ValueError(
            "paired error arrays differ in shape: "
            f"{original_error.shape} vs {intervention_error.shape}"
        )
    if original_error.size == 0:
        raise ValueError("paired error arrays are empty")

    paired_delta = intervention_error - original_error
    rng = np.random.default_rng(seed)
    n = paired_delta.size
    bootstrapped = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        bootstrapped[index] = paired_delta[sample].mean()

    lower, upper = np.quantile(bootstrapped, [0.025, 0.975])
    probability_nonpositive = (
        np.count_nonzero(bootstrapped <= 0.0) + 1
    ) / (n_bootstrap + 1)
    probability_nonnegative = (
        np.count_nonzero(bootstrapped >= 0.0) + 1
    ) / (n_bootstrap + 1)
    p_value = min(
        1.0,
        2.0 * min(probability_nonpositive, probability_nonnegative),
    )
    return {
        "n": int(n),
        "original_mae": float(original_error.mean()),
        "intervention_mae": float(intervention_error.mean()),
        "delta_mae_intervention_minus_original": float(paired_delta.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "paired_bootstrap_p_two_sided": float(p_value),
    }


def summarize(
    path: str,
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, object]:
    """Read a long-form intervention dump and compare every mode to original."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"confidence intervention CSV contains no rows: {path}")

    required = {"dataset", "mode", "sample_index", "true_label", "prediction"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    dataset_names = {row["dataset"] for row in rows}
    if len(dataset_names) != 1:
        raise ValueError(f"{path} contains multiple datasets: {sorted(dataset_names)}")
    dataset = next(iter(dataset_names))

    by_mode: dict[str, dict[int, tuple[float, float]]] = {}
    for row in rows:
        mode = row["mode"]
        sample_index = int(row["sample_index"])
        label = float(row["true_label"])
        prediction = float(row["prediction"])
        if not np.isfinite(label) or not np.isfinite(prediction):
            raise ValueError(f"{path} contains non-finite labels or predictions")
        mode_rows = by_mode.setdefault(mode, {})
        if sample_index in mode_rows:
            raise ValueError(f"{path}: duplicate {mode=} {sample_index=}")
        mode_rows[sample_index] = (label, prediction)

    if "original" not in by_mode:
        raise ValueError(f"{path} has no original mode")
    sample_indices = sorted(by_mode["original"])
    reference_labels = np.asarray(
        [by_mode["original"][index][0] for index in sample_indices],
        dtype=np.float64,
    )
    original_predictions = np.asarray(
        [by_mode["original"][index][1] for index in sample_indices],
        dtype=np.float64,
    )
    original_error = np.abs(original_predictions - reference_labels)

    comparisons: dict[str, object] = {}
    for mode, mode_rows in by_mode.items():
        if mode == "original":
            continue
        if sorted(mode_rows) != sample_indices:
            raise ValueError(f"{path}: sample IDs differ for mode {mode!r}")
        labels = np.asarray(
            [mode_rows[index][0] for index in sample_indices],
            dtype=np.float64,
        )
        if not np.array_equal(labels, reference_labels):
            raise ValueError(f"{path}: labels differ for mode {mode!r}")
        predictions = np.asarray(
            [mode_rows[index][1] for index in sample_indices],
            dtype=np.float64,
        )
        comparisons[mode] = _paired_mae_bootstrap(
            original_error,
            np.abs(predictions - labels),
            seed=seed,
            n_bootstrap=n_bootstrap,
        )

    return {
        "dataset": dataset,
        "source": os.path.abspath(path),
        "difference_definition": "MAE(intervention) - MAE(original)",
        "positive_delta_means": "intervention is worse",
        "bootstrap_seed": seed,
        "n_bootstrap": n_bootstrap,
        "comparisons": comparisons,
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
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be positive")

    payload: dict[str, object] = {}
    for item in args.input:
        if "=" not in item:
            raise SystemExit(f"invalid --input {item!r}; expected DATASET=CSV")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit("dataset name cannot be empty")
        if name in payload:
            raise SystemExit(f"duplicate dataset: {name}")
        summary = summarize(
            os.path.abspath(path),
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
        )
        if summary["dataset"] != name:
            raise SystemExit(
                f"dataset key {name!r} does not match CSV dataset "
                f"{summary['dataset']!r}"
            )
        payload[name] = summary

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {output}")
    for dataset, summary in payload.items():
        comparisons = summary["comparisons"]
        for mode, result in comparisons.items():
            print(
                f"{dataset}/{mode}: "
                f"delta_MAE={result['delta_mae_intervention_minus_original']:+.6f} "
                f"95% CI [{result['ci95_lower']:+.6f}, "
                f"{result['ci95_upper']:+.6f}], "
                f"p={result['paired_bootstrap_p_two_sided']:.6g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
