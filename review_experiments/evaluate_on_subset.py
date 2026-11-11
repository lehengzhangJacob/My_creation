#!/usr/bin/env python3
"""Subset MAE/Corr/Acc-2 and a paired bootstrap against one baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from paired_bootstrap import (
    PredictionData,
    load_prediction_csv,
    paired_bootstrap,
    validate_pair,
)


def _corr(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size < 2 or float(np.std(labels)) == 0.0 or float(np.std(preds)) == 0.0:
        return float("nan")
    return float(np.corrcoef(labels, preds)[0, 1])


def _acc2(labels: np.ndarray, preds: np.ndarray) -> float:
    keep = np.abs(labels) > 1e-8
    if not np.any(keep):
        return float("nan")
    return float(np.mean(np.sign(labels[keep]) == np.sign(preds[keep])))


def regression_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "n": int(labels.size),
        "mae": float(np.mean(np.abs(labels - preds))) if labels.size else float("nan"),
        "corr": _corr(labels, preds),
        "acc2": _acc2(labels, preds),
    }


def load_conflict_index(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    flags = np.zeros(len(rows), dtype=bool)
    for row in rows:
        flags[int(row["global_index"])] = bool(int(row["is_conflict"]))
    return flags


def subset_metrics(labels: np.ndarray, preds: np.ndarray, conflict: np.ndarray) -> dict:
    if labels.shape != preds.shape or labels.shape != conflict.shape:
        raise ValueError(
            f"shape mismatch labels={labels.shape} preds={preds.shape} conflict={conflict.shape}"
        )
    return {
        "conflict": regression_metrics(labels[conflict], preds[conflict]),
        "nonconflict": regression_metrics(labels[~conflict], preds[~conflict]),
        "all": regression_metrics(labels, preds),
    }


def _slice(data: PredictionData, mask: np.ndarray) -> PredictionData:
    return PredictionData(
        path=data.path,
        sample_ids=data.sample_ids[mask],
        labels=data.labels[mask],
        predictions=data.predictions[mask],
        prediction_source=data.prediction_source,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=128)
    args = parser.parse_args()
    conflict = load_conflict_index(args.subset)
    pred = load_prediction_csv(args.predictions, "regression")
    if pred.labels.shape != conflict.shape:
        raise ValueError(f"{args.predictions} rows do not match the subset")
    report = {
        "predictions": str(args.predictions),
        "subset": str(args.subset),
        "metrics": subset_metrics(pred.labels, pred.predictions, conflict),
    }
    if args.baseline is not None:
        base = load_prediction_csv(args.baseline, "regression")
        validate_pair(pred, base)
        report["baseline"] = str(args.baseline)
        report["baseline_metrics"] = subset_metrics(base.labels, base.predictions, conflict)
        report["bootstrap_mae_conflict"] = paired_bootstrap(
            _slice(pred, conflict),
            _slice(base, conflict),
            metric="mae",
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            confidence=0.95,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
