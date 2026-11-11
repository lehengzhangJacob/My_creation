#!/usr/bin/env python3
"""Collect development-selected test results from checkpoints or logs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from review_common import (
    DEFAULT_MANIFEST,
    HARNESS_DIR,
    atomic_write_json,
    atomic_write_text,
    expected_artifact_paths,
    load_manifest,
    resolve_my_creation_path,
    select_experiments,
    utc_now,
)


BEST_HEADER_RE = re.compile(
    r"^Best Results \((?P<selection_metric>[^,\)]+),[^\)]*\):\s*$",
    re.MULTILINE,
)
BLOCK_END_RE = re.compile(
    r"^\s*(?:Last Epoch|Best Test\b).*?$", re.MULTILINE
)
SELECTION_RE = re.compile(
    r"^\s*Selection score(?: \(best compare\))?:\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)
METRIC_RE = re.compile(
    r"^\s*(?P<name>Acc(?:-[2357])?|F1|MAE|Corr):\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?P<percent>%?)\s*$",
    re.MULTILINE,
)
EPOCH_RE = re.compile(r"^\s*Epoch\s+(?P<epoch>\d+)/\d+", re.MULTILINE)
SAVE_RE = re.compile(r"^\s*>> Best model saved\b", re.MULTILINE)

METRIC_ALIASES = {
    "acc": "acc2",
    "acc-2": "acc2",
    "acc-3": "acc3",
    "acc-5": "acc5",
    "acc-7": "acc7",
    "f1": "f1",
    "mae": "mae",
    "corr": "corr",
}
CSV_METRICS = ("acc2", "acc3", "acc5", "acc7", "f1", "mae", "corr")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=HARNESS_DIR / "results")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Same ID glob/dataset/family/tag selector as the launcher.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero when any selected result is unavailable or invalid.",
    )
    parser.add_argument(
        "--no-export-predictions",
        action="store_true",
        help="Do not export checkpoint test_preds/test_labels to normalized CSV.",
    )
    parser.add_argument(
        "--logs-only",
        action="store_true",
        help="Skip checkpoint loading and parse only the Best Results log block.",
    )
    parser.add_argument(
        "--artifact",
        "--checkpoint",
        dest="checkpoint_overrides",
        action="append",
        default=[],
        metavar="EXPERIMENT_ID=PATH",
        help=(
            "Import an explicit checkpoint file or run/checkpoint directory. "
            "Repeat for externally managed experiments."
        ),
    )
    parser.add_argument(
        "--log",
        dest="log_overrides",
        action="append",
        default=[],
        metavar="EXPERIMENT_ID=PATH",
        help="Import an explicit training log for an experiment.",
    )
    parser.add_argument(
        "--predictions",
        dest="prediction_overrides",
        action="append",
        default=[],
        metavar="EXPERIMENT_ID=PATH",
        help="Associate an explicit best_test_predictions.csv with an experiment.",
    )
    return parser.parse_args(argv)


def _finite_float(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {result}")
    return result


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint root is {type(checkpoint).__name__}, not dict")
    return checkpoint


def _normalize_test_results(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise TypeError("checkpoint test_results is not a dictionary")
    metrics: dict[str, float] = {}
    for raw_name, raw_value in raw.items():
        name = METRIC_ALIASES.get(str(raw_name).lower(), str(raw_name).lower())
        if name == "epoch":
            continue
        try:
            metrics[name] = _finite_float(raw_value, f"test_results.{raw_name}")
        except (TypeError, ValueError):
            continue
    if not metrics:
        raise ValueError("checkpoint test_results contains no numeric metrics")
    return metrics


def _write_prediction_csv(
    experiment: dict[str, Any],
    predictions: Any,
    labels: Any,
    output_path: Path,
) -> int:
    pred_values = np.asarray(predictions).reshape(-1)
    label_values = np.asarray(labels).reshape(-1)
    if pred_values.shape != label_values.shape:
        raise ValueError(
            f"checkpoint test pred/label shape mismatch: "
            f"{pred_values.shape} vs {label_values.shape}"
        )
    if pred_values.size == 0:
        raise ValueError("checkpoint test predictions are empty")
    if not np.all(np.isfinite(pred_values)) or not np.all(np.isfinite(label_values)):
        raise ValueError("checkpoint test predictions/labels contain non-finite values")

    buffer = io.StringIO(newline="")
    if experiment["task"] == "binary":
        clipped_logits = np.clip(pred_values.astype(float), -709.0, 709.0)
        probabilities = 1.0 / (1.0 + np.exp(-clipped_logits))
        classes = (probabilities >= 0.5).astype(int)
        writer = csv.writer(buffer)
        writer.writerow(
            ["sample_index", "true_label", "logit", "probability", "prediction"]
        )
        for index, (label, logit, probability, prediction) in enumerate(
            zip(label_values, pred_values, probabilities, classes)
        ):
            writer.writerow(
                [
                    index,
                    int(label),
                    float(logit),
                    float(probability),
                    int(prediction),
                ]
            )
    else:
        writer = csv.writer(buffer)
        writer.writerow(["sample_index", "true_label", "prediction"])
        for index, (label, prediction) in enumerate(zip(label_values, pred_values)):
            writer.writerow([index, float(label), float(prediction)])
    atomic_write_text(output_path, buffer.getvalue())
    return int(pred_values.size)


def parse_dev_selected_log(text: str) -> dict[str, Any]:
    """Parse only the final ``Best Results`` block, never the oracle block."""
    matches = list(BEST_HEADER_RE.finditer(text))
    if not matches:
        raise ValueError("no complete 'Best Results (...)' dev-selected block")
    header = matches[-1]
    tail = text[header.end() :]
    end_match = BLOCK_END_RE.search(tail)
    block = tail[: end_match.start()] if end_match else tail

    selection_match = SELECTION_RE.search(block)
    if not selection_match:
        raise ValueError("dev-selected block has no selection score")
    selection_score = _finite_float(
        selection_match.group("value"), "selection_score"
    )

    metrics: dict[str, float] = {}
    for metric_match in METRIC_RE.finditer(block):
        raw_name = metric_match.group("name").lower()
        name = METRIC_ALIASES[raw_name]
        value = _finite_float(metric_match.group("value"), name)
        if metric_match.group("percent"):
            value /= 100.0
        metrics[name] = value
    if not metrics:
        raise ValueError("dev-selected block has no test metrics")

    selected_epoch: int | None = None
    prefix_start = text.rfind("InfoGate Training", 0, header.start())
    classify_start = text.rfind(
        "InfoGate Binary Classification Training", 0, header.start()
    )
    segment_start = max(prefix_start, classify_start, 0)
    segment = text[segment_start : header.start()]
    current_epoch: int | None = None
    events = sorted(
        [
            *((match.start(), "epoch", match) for match in EPOCH_RE.finditer(segment)),
            *((match.start(), "save", match) for match in SAVE_RE.finditer(segment)),
        ],
        key=lambda item: item[0],
    )
    for _, event_type, event_match in events:
        if event_type == "epoch":
            current_epoch = int(event_match.group("epoch"))
        elif current_epoch is not None:
            selected_epoch = current_epoch

    return {
        "selection_metric": header.group("selection_metric").strip(),
        "selection_score": selection_score,
        "epoch": selected_epoch,
        "test_results": metrics,
    }


def _parse_path_overrides(
    values: list[str],
    *,
    experiments: dict[str, dict[str, Any]],
    kind: str,
) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"{kind} import must use EXPERIMENT_ID=PATH, got {value!r}"
            )
        experiment_id, raw_path = value.split("=", 1)
        experiment_id = experiment_id.strip()
        raw_path = raw_path.strip()
        if experiment_id not in experiments:
            raise ValueError(
                f"{kind} import names unknown/unselected experiment "
                f"{experiment_id!r}"
            )
        if not raw_path:
            raise ValueError(f"{kind} import for {experiment_id} has an empty path")
        if experiment_id in parsed:
            raise ValueError(
                f"duplicate {kind} import for experiment {experiment_id}"
            )
        path = Path(raw_path).expanduser().resolve()
        if kind == "checkpoint" and path.is_dir():
            checkpoint_name = experiments[experiment_id].get(
                "expected_checkpoint",
                f"infogate_{experiments[experiment_id]['dataset']}_best.pt",
            )
            candidates = [
                path / checkpoint_name,
                path / "checkpoints" / checkpoint_name,
            ]
            path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                candidates[0],
            )
        parsed[experiment_id] = path
    return parsed


def _experiment_paths(
    experiment: dict[str, Any],
    *,
    checkpoint_overrides: dict[str, Path],
    log_overrides: dict[str, Path],
    prediction_overrides: dict[str, Path],
) -> tuple[Path | None, Path | None, Path | None]:
    experiment_id = experiment["id"]
    explicit_checkpoint = checkpoint_overrides.get(experiment_id)
    explicit_log = log_overrides.get(experiment_id)
    explicit_predictions = prediction_overrides.get(experiment_id)
    if experiment["action"] == "train":
        checkpoint_path, generated_predictions = expected_artifact_paths(experiment)
        return (
            explicit_checkpoint or checkpoint_path,
            explicit_log or HARNESS_DIR / "logs" / f"{experiment_id}.log",
            explicit_predictions or generated_predictions,
        )

    source = experiment.get("source", {})
    checkpoint_path = (
        resolve_my_creation_path(source["checkpoint"])
        if source.get("checkpoint")
        else None
    )
    log_path = (
        resolve_my_creation_path(source["log"]) if source.get("log") else None
    )
    predictions_path = (
        resolve_my_creation_path(source["predictions"])
        if source.get("predictions")
        else None
    )
    return (
        explicit_checkpoint or checkpoint_path,
        explicit_log or log_path,
        explicit_predictions or predictions_path,
    )


def collect_one(
    experiment: dict[str, Any],
    *,
    predictions_root: Path,
    export_predictions: bool,
    checkpoint_overrides: dict[str, Path] | None = None,
    log_overrides: dict[str, Path] | None = None,
    prediction_overrides: dict[str, Path] | None = None,
    logs_only: bool = False,
) -> dict[str, Any]:
    checkpoint_overrides = checkpoint_overrides or {}
    log_overrides = log_overrides or {}
    prediction_overrides = prediction_overrides or {}
    checkpoint_path, log_path, existing_predictions = _experiment_paths(
        experiment,
        checkpoint_overrides=checkpoint_overrides,
        log_overrides=log_overrides,
        prediction_overrides=prediction_overrides,
    )
    result: dict[str, Any] = {
        "experiment_id": experiment["id"],
        "dataset": experiment["dataset"],
        "task": experiment["task"],
        "family": experiment.get("family"),
        "condition": experiment.get("condition"),
        "seed": experiment.get("seed"),
        "action": experiment["action"],
        "status": "missing",
        "headline": "development-selected checkpoint test result",
        "test_oracle": False,
        "selection_metric": None,
        "selection_score": None,
        "epoch": None,
        "test_results": {},
        "args": None,
        "predictions": {
            "available": bool(
                existing_predictions and existing_predictions.is_file()
            ),
            "path": str(existing_predictions) if existing_predictions else None,
            "n": None,
        },
        "provenance": {
            "source": None,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "log": str(log_path) if log_path else None,
            "explicit_import": experiment["id"]
            in {
                *checkpoint_overrides,
                *log_overrides,
                *prediction_overrides,
            },
        },
        "warnings": [],
    }

    if (not logs_only) and checkpoint_path and checkpoint_path.is_file():
        try:
            checkpoint = _torch_load(checkpoint_path)
            result["test_results"] = _normalize_test_results(
                checkpoint.get("test_results")
            )
            result["selection_score"] = _finite_float(
                checkpoint.get("selection_score"), "selection_score"
            )
            result["selection_metric"] = str(
                checkpoint.get("selection_metric", "unknown")
            )
            if checkpoint.get("epoch") is not None:
                result["epoch"] = int(checkpoint["epoch"])
            result["args"] = _jsonable(checkpoint.get("args"))

            test_predictions = checkpoint.get("test_preds")
            test_labels = checkpoint.get("test_labels")
            if test_predictions is not None or test_labels is not None:
                if test_predictions is None or test_labels is None:
                    raise ValueError(
                        "checkpoint must contain both test_preds and test_labels"
                    )
                if export_predictions:
                    normalized_path = (
                        predictions_root
                        / experiment["id"]
                        / "best_test_predictions.csv"
                    )
                    count = _write_prediction_csv(
                        experiment,
                        test_predictions,
                        test_labels,
                        normalized_path,
                    )
                    result["predictions"] = {
                        "available": True,
                        "path": str(normalized_path),
                        "n": count,
                    }
                else:
                    pred_values = np.asarray(test_predictions).reshape(-1)
                    label_values = np.asarray(test_labels).reshape(-1)
                    if pred_values.shape != label_values.shape:
                        raise ValueError(
                            "checkpoint test pred/label shape mismatch: "
                            f"{pred_values.shape} vs {label_values.shape}"
                        )
                    result["predictions"]["n"] = int(pred_values.size)
            result["status"] = "complete"
            result["provenance"]["source"] = "checkpoint"
            return result
        except Exception as exc:
            result["warnings"].append(
                f"checkpoint parse failed; attempted log fallback: "
                f"{type(exc).__name__}: {exc}"
            )

    if log_path and log_path.is_file():
        try:
            parsed = parse_dev_selected_log(
                log_path.read_text(encoding="utf-8", errors="replace")
            )
            result.update(parsed)
            result["status"] = "complete"
            result["provenance"]["source"] = "log"
            return result
        except Exception as exc:
            result["status"] = "invalid"
            result["warnings"].append(
                f"log parse failed: {type(exc).__name__}: {exc}"
            )
            return result

    result["warnings"].append(
        "no checkpoint or log is available; this experiment has not been collected"
    )
    return result


def _csv_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "experiment_id": result["experiment_id"],
            "dataset": result["dataset"],
            "task": result["task"],
            "family": result["family"],
            "condition": result["condition"],
            "seed": result["seed"],
            "action": result["action"],
            "status": result["status"],
            "source": result["provenance"]["source"],
            "selection_metric": result["selection_metric"],
            "selection_score": result["selection_score"],
            "selected_epoch": result["epoch"],
            "headline": result["headline"],
            "test_oracle": result["test_oracle"],
            "checkpoint_path": result["provenance"]["checkpoint"],
            "log_path": result["provenance"]["log"],
            "predictions_path": result["predictions"]["path"],
            "prediction_count": result["predictions"]["n"],
            "warnings": " | ".join(result["warnings"]),
        }
        for metric in CSV_METRICS:
            row[metric] = result["test_results"].get(metric)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "experiment_id",
        "dataset",
        "task",
        "family",
        "condition",
        "seed",
        "action",
        "status",
        "source",
        "selection_metric",
        "selection_score",
        "selected_epoch",
        *CSV_METRICS,
        "headline",
        "test_oracle",
        "checkpoint_path",
        "log_path",
        "predictions_path",
        "prediction_count",
        "warnings",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    selected = select_experiments(manifest["experiments"], args.only)
    selected_by_id = {experiment["id"]: experiment for experiment in selected}
    checkpoint_overrides = _parse_path_overrides(
        args.checkpoint_overrides,
        experiments=selected_by_id,
        kind="checkpoint",
    )
    log_overrides = _parse_path_overrides(
        args.log_overrides,
        experiments=selected_by_id,
        kind="log",
    )
    prediction_overrides = _parse_path_overrides(
        args.prediction_overrides,
        experiments=selected_by_id,
        kind="predictions",
    )
    output_dir = args.output_dir.expanduser().resolve()
    predictions_root = output_dir / "predictions"

    results = [
        collect_one(
            experiment,
            predictions_root=predictions_root,
            export_predictions=not args.no_export_predictions,
            checkpoint_overrides=checkpoint_overrides,
            log_overrides=log_overrides,
            prediction_overrides=prediction_overrides,
            logs_only=args.logs_only,
        )
        for experiment in selected
    ]
    counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("complete", "missing", "invalid")
    }
    document = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "manifest": str(args.manifest.expanduser().resolve()),
        "headline_policy": manifest["headline_policy"],
        "counts": counts,
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    atomic_write_json(json_path, document)
    _write_csv(csv_path, _csv_rows(results))
    print(
        f"Wrote {len(results)} results ({counts['complete']} complete, "
        f"{counts['missing']} missing, {counts['invalid']} invalid)\n"
        f"  JSON: {json_path}\n"
        f"  CSV:  {csv_path}"
    )
    unavailable = counts["missing"] + counts["invalid"]
    return 1 if args.strict and unavailable else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
