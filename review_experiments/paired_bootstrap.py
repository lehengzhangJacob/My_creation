#!/usr/bin/env python3
"""Paired bootstrap for two same-order best-test prediction CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from review_common import atomic_write_json, utc_now


ID_COLUMNS = ("sample_index", "sample_id", "id")
LABEL_COLUMNS = ("true_label", "label", "y")
REGRESSION_PREDICTION_COLUMNS = ("prediction", "pred", "y_pred")


@dataclass(frozen=True)
class PredictionData:
    path: Path
    sample_ids: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    prediction_source: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="First prediction CSV.")
    parser.add_argument("--b", type=Path, required=True, help="Second prediction CSV.")
    parser.add_argument("--task", choices=("regression", "binary"), required=True)
    parser.add_argument(
        "--metric",
        default=None,
        help="Regression: mae/rmse. Binary: accuracy/f1 (default: mae/accuracy).",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--name-a", default="a")
    parser.add_argument("--name-b", default="b")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be positive")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.metric is None:
        args.metric = "mae" if args.task == "regression" else "accuracy"
    choices = {
        "regression": {"mae", "rmse"},
        "binary": {"accuracy", "f1"},
    }
    if args.metric not in choices[args.task]:
        parser.error(
            f"--metric {args.metric!r} is invalid for {args.task}; "
            f"choose from {sorted(choices[args.task])}"
        )
    return args


def _find_column(fieldnames: list[str], candidates: tuple[str, ...], role: str) -> str:
    lowered = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        f"missing {role} column; expected one of {', '.join(candidates)}"
    )


def _finite_array(values: list[float], role: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.size == 0:
        raise ValueError(f"{role} is empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{role} contains non-finite values")
    return result


def load_prediction_csv(path: Path, task: str) -> PredictionData:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{resolved} does not exist. Paired bootstrap requires per-sample "
            "predictions; an external baseline with aggregate metrics only "
            "cannot be tested."
        )
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(
                f"{resolved}: no CSV header; aggregate/external baselines without "
                "per-sample predictions are not eligible"
            )
        fieldnames = list(reader.fieldnames)
        id_column = _find_column(fieldnames, ID_COLUMNS, "sample ID")
        label_column = _find_column(fieldnames, LABEL_COLUMNS, "true-label")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{resolved}: prediction CSV has no samples")

    sample_ids = np.asarray([row[id_column] for row in rows], dtype=str)
    if np.any(sample_ids == ""):
        raise ValueError(f"{resolved}: sample IDs must not be empty")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError(f"{resolved}: sample IDs must be unique")
    try:
        labels = _finite_array(
            [float(row[label_column]) for row in rows], f"{resolved}: labels"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{resolved}: invalid true-label values: {exc}") from exc

    lowered = {name.strip().lower(): name for name in fieldnames}
    if task == "regression":
        prediction_column = _find_column(
            fieldnames, REGRESSION_PREDICTION_COLUMNS, "prediction"
        )
        try:
            predictions = _finite_array(
                [float(row[prediction_column]) for row in rows],
                f"{resolved}: predictions",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{resolved}: invalid predictions: {exc}") from exc
        prediction_source = prediction_column
    else:
        if not np.all(np.equal(labels, labels.astype(int))) or not set(
            labels.astype(int).tolist()
        ).issubset({0, 1}):
            raise ValueError(f"{resolved}: binary labels must be exactly 0/1")
        if "prediction" in lowered:
            prediction_column = lowered["prediction"]
            try:
                predictions = _finite_array(
                    [float(row[prediction_column]) for row in rows],
                    f"{resolved}: predictions",
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{resolved}: invalid predictions: {exc}") from exc
            if not np.all(np.equal(predictions, predictions.astype(int))) or not set(
                predictions.astype(int).tolist()
            ).issubset({0, 1}):
                raise ValueError(
                    f"{resolved}: binary prediction column must contain 0/1"
                )
            predictions = predictions.astype(int)
            prediction_source = prediction_column
        elif "probability" in lowered:
            probability_column = lowered["probability"]
            probabilities = _finite_array(
                [float(row[probability_column]) for row in rows],
                f"{resolved}: probabilities",
            )
            if np.any((probabilities < 0.0) | (probabilities > 1.0)):
                raise ValueError(f"{resolved}: probabilities must be within [0, 1]")
            predictions = (probabilities >= 0.5).astype(int)
            prediction_source = f"{probability_column}>=0.5"
        elif "logit" in lowered:
            logit_column = lowered["logit"]
            logits = _finite_array(
                [float(row[logit_column]) for row in rows],
                f"{resolved}: logits",
            )
            predictions = (logits >= 0.0).astype(int)
            prediction_source = f"{logit_column}>=0"
        else:
            raise ValueError(
                f"{resolved}: missing binary prediction/probability/logit column. "
                "Aggregate/external baselines without per-sample predictions "
                "cannot be tested."
            )

    return PredictionData(
        path=resolved,
        sample_ids=sample_ids,
        labels=labels,
        predictions=np.asarray(predictions),
        prediction_source=prediction_source,
    )


def validate_pair(a: PredictionData, b: PredictionData) -> None:
    if len(a.labels) != len(b.labels):
        raise ValueError(
            f"sample length mismatch: {a.path} has {len(a.labels)}, "
            f"{b.path} has {len(b.labels)}"
        )
    if not np.array_equal(a.sample_ids, b.sample_ids):
        mismatch = np.flatnonzero(a.sample_ids != b.sample_ids)
        first = int(mismatch[0]) if mismatch.size else -1
        raise ValueError(
            f"sample ID order mismatch at row {first}: "
            f"{a.sample_ids[first]!r} != {b.sample_ids[first]!r}"
        )
    if not np.array_equal(a.labels, b.labels):
        mismatch = np.flatnonzero(a.labels != b.labels)
        first = int(mismatch[0]) if mismatch.size else -1
        raise ValueError(
            f"label order/value mismatch at row {first}: "
            f"{a.labels[first]!r} != {b.labels[first]!r}"
        )


def _mae(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - labels)))


def _rmse(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predictions - labels))))


def _accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(predictions.astype(int) == labels.astype(int)))


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels_i = labels.astype(int)
    predictions_i = predictions.astype(int)
    weighted_sum = 0.0
    for class_id in (0, 1):
        support = int(np.sum(labels_i == class_id))
        if support == 0:
            continue
        tp = int(np.sum((labels_i == class_id) & (predictions_i == class_id)))
        fp = int(np.sum((labels_i != class_id) & (predictions_i == class_id)))
        fn = int(np.sum((labels_i == class_id) & (predictions_i != class_id)))
        denominator = 2 * tp + fp + fn
        class_f1 = float(2 * tp / denominator) if denominator else 0.0
        weighted_sum += support * class_f1
    return weighted_sum / len(labels_i)


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mae": _mae,
    "rmse": _rmse,
    "accuracy": _accuracy,
    "f1": _f1,
}


def paired_bootstrap(
    a: PredictionData,
    b: PredictionData,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> dict[str, float | int]:
    validate_pair(a, b)
    score = METRICS[metric]
    estimate_a = score(a.labels, a.predictions)
    estimate_b = score(b.labels, b.predictions)
    observed_difference = estimate_a - estimate_b

    rng = np.random.default_rng(seed)
    n_samples = len(a.labels)
    differences = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n_samples, size=n_samples)
        differences[index] = score(
            a.labels[sample], a.predictions[sample]
        ) - score(b.labels[sample], b.predictions[sample])
    if not np.all(np.isfinite(differences)):
        raise ValueError("bootstrap produced non-finite differences")

    alpha = 1.0 - confidence
    lower, upper = np.quantile(differences, [alpha / 2.0, 1.0 - alpha / 2.0])
    probability_nonpositive = (
        np.count_nonzero(differences <= 0.0) + 1
    ) / (n_bootstrap + 1)
    probability_nonnegative = (
        np.count_nonzero(differences >= 0.0) + 1
    ) / (n_bootstrap + 1)
    p_value = min(
        1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)
    )
    return {
        "n_samples": n_samples,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "confidence": confidence,
        "estimate_a": estimate_a,
        "estimate_b": estimate_b,
        "difference_a_minus_b": observed_difference,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value_two_sided": float(p_value),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    a = load_prediction_csv(args.a, args.task)
    b = load_prediction_csv(args.b, args.task)
    statistics = paired_bootstrap(
        a,
        b,
        metric=args.metric,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        confidence=args.confidence,
    )
    direction = (
        "lower_is_better" if args.metric in {"mae", "rmse"} else "higher_is_better"
    )
    output = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "test": "paired_percentile_bootstrap",
        "task": args.task,
        "metric": args.metric,
        "metric_direction": direction,
        "difference_definition": "metric(a) - metric(b)",
        "a": {
            "name": args.name_a,
            "path": str(a.path),
            "prediction_source": a.prediction_source,
        },
        "b": {
            "name": args.name_b,
            "path": str(b.path),
            "prediction_source": b.prediction_source,
        },
        **statistics,
    }
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        atomic_write_json(output_path, output)
        print(f"Wrote {output_path}", file=sys.stderr)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
