#!/usr/bin/env python3
"""MOSEI DeBERTa trial 36 hyper-parameter sensitivity (paper Fig. mosei_sensitivity).

Freezes all knobs to the development-selected Nov-2026 search trial 36
(``sota_search/runs/mosei_deberta/trial_36`` / paper PRISM DeBERTa MOSEI row),
then varies one of ``beta_ib``, ``alpha_ib``, ``mse_weight``,
``selector_target_temp``, ``selector_rib_weight`` within Table
``tab:hyperparam_search_space`` ranges.

Uses ``ablation_study/train.py`` (same trainer as the search). The exact
default (center) point is not re-trained: inject trial-36 Best Results when
aggregating / seeding.

Commands:
  list-values --axis NAME
  train --axis NAME --value FLOAT [--dry-run]
  aggregate --runs-root DIR [--summary-out PATH] [--inject-center]
  queue-plan                 Print (axis, value) jobs excluding the center.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

_MY = Path(__file__).resolve().parent.parent
_TRAIN = _MY / "ablation_study" / "train.py"

# Float formatting mirrors sota_search / trial_36 cmd.json.
TRIAL36 = {
    "n_epochs": 50,
    "train_batch_size": 4,
    "gradient_accumulation_step": 8,
    "selection_metric": "mae",
    "seed": 4861,
    "learning_rate": "2.047660e-05",
    "ig_learning_rate": "4.221805e-04",
    "beta_ib": 19.2729,
    "num_infogate_layers": 2,
    "bottleneck_dim": 64,
    "mse_weight": 2.2427,
    "dropout_prob": "0.2530",
    "alpha_ib": 0.009207,
    "stage1_epochs": 11,
    "warmup_proportion": "0.1156",
    "weight_decay": "0.000948",
    "ema_decay": "0.999",
    "selector_target_temp": 0.4076,
    "selector_rib_weight": 0.0420,
    "gumbel_tau_start": "0.9798",
    "gumbel_tau_end": "0.4381",
    "num_heads": 4,
    "unified_dim": 256,
    "ib_hidden_dim": 256,
    "ema_start_epoch": 5,
    "early_stop_patience": 10,
}

# Gold test metrics at the frozen center (trial 36 Best Results).
CENTER_METRICS = {
    "MAE": 0.4988,
    "Corr": 0.8035,
    "F1": 0.8737,
    "Acc-2": 0.8740,
    "Acc-7": 0.5624,
    "SelectionScore": 0.488377,
}

SUPPORTED_AXES = (
    "beta_ib",
    "mse_weight",
    "alpha_ib",
    "selector_target_temp",
    "selector_rib_weight",
)

RESULT_LINE_RE = re.compile(
    r"\s+(Selection score|Acc-2|Acc-7|Acc-5|Acc-3|MAE|Corr|F1):\s+([\d.]+)%?"
)

# Table 2 ranges; grids always include the trial-36 center.
def default_grid(axis: str) -> list[float]:
    b0 = float(TRIAL36["beta_ib"])
    m0 = float(TRIAL36["mse_weight"])
    a0 = float(TRIAL36["alpha_ib"])
    t0 = float(TRIAL36["selector_target_temp"])
    w0 = float(TRIAL36["selector_rib_weight"])
    if axis == "beta_ib":
        # Table: [5, 25]
        return sorted({5.0, 10.0, 15.0, b0, 22.5, 25.0})
    if axis == "mse_weight":
        # Table: [0.5, 2.5]
        return sorted({0.5, 1.0, 1.5, 2.0, m0, 2.5})
    if axis == "alpha_ib":
        # Table: [1e-4, 1e-2]
        return sorted({1e-4, 5e-4, 1e-3, 5e-3, a0, 1e-2})
    if axis == "selector_target_temp":
        # Table: [0.3, 0.9]
        return sorted({0.3, t0, 0.5, 0.7, 0.9})
    if axis == "selector_rib_weight":
        # Table: [0.04, 0.15]
        return sorted({0.04, w0, 0.06, 0.08, 0.10, 0.15})
    raise ValueError(f"unsupported axis: {axis}")


def is_center(axis: str, value: float, tol: float = 1e-9) -> bool:
    return abs(float(value) - float(TRIAL36[axis])) <= tol * max(1.0, abs(float(TRIAL36[axis])))


def value_tag(axis: str, value: float) -> str:
    if axis == "alpha_ib":
        s = f"{float(value):.6f}".rstrip("0").rstrip(".")
    else:
        s = f"{float(value):.6g}"
    return f"v{s.replace('.', 'p').replace('-', 'm')}"


def _fmt_override(axis: str, value: float) -> str:
    if axis == "alpha_ib":
        return f"{float(value):.6f}"
    if axis == "beta_ib":
        return f"{float(value):.4f}"
    if axis in ("mse_weight", "selector_target_temp", "selector_rib_weight"):
        return f"{float(value):.4f}"
    return str(value)


def build_train_argv(*, checkpoint_dir: str, overrides: dict[str, float]) -> list[str]:
    p = dict(TRIAL36)
    for k, v in overrides.items():
        p[k] = v
    argv = [
        "--dataset", "mosei",
        "--n_epochs", str(int(p["n_epochs"])),
        "--train_batch_size", str(int(p["train_batch_size"])),
        "--gradient_accumulation_step", str(int(p["gradient_accumulation_step"])),
        "--checkpoint_dir", checkpoint_dir,
        "--selection_metric", str(p["selection_metric"]),
        "--seed", str(int(p["seed"])),
        "--learning_rate", str(p["learning_rate"]),
        "--ig_learning_rate", str(p["ig_learning_rate"]),
        "--beta_ib", _fmt_override("beta_ib", float(p["beta_ib"])) if not isinstance(p["beta_ib"], str) else p["beta_ib"],
        "--num_infogate_layers", str(int(p["num_infogate_layers"])),
        "--bottleneck_dim", str(int(p["bottleneck_dim"])),
        "--mse_weight", _fmt_override("mse_weight", float(p["mse_weight"])) if not isinstance(p["mse_weight"], str) else p["mse_weight"],
        "--dropout_prob", str(p["dropout_prob"]),
        "--alpha_ib", _fmt_override("alpha_ib", float(p["alpha_ib"])) if not isinstance(p["alpha_ib"], str) else p["alpha_ib"],
        "--stage1_epochs", str(int(p["stage1_epochs"])),
        "--warmup_proportion", str(p["warmup_proportion"]),
        "--weight_decay", str(p["weight_decay"]),
        "--ema_decay", str(p["ema_decay"]),
        "--selector_target_temp",
        _fmt_override("selector_target_temp", float(p["selector_target_temp"]))
        if not isinstance(p["selector_target_temp"], str) else p["selector_target_temp"],
        "--selector_rib_weight",
        _fmt_override("selector_rib_weight", float(p["selector_rib_weight"]))
        if not isinstance(p["selector_rib_weight"], str) else p["selector_rib_weight"],
        "--gumbel_tau_start", str(p["gumbel_tau_start"]),
        "--gumbel_tau_end", str(p["gumbel_tau_end"]),
        "--num_heads", str(int(p["num_heads"])),
        "--unified_dim", str(int(p["unified_dim"])),
        "--ib_hidden_dim", str(int(p["ib_hidden_dim"])),
        "--ema_start_epoch", str(int(p["ema_start_epoch"])),
        "--early_stop_patience", str(int(p["early_stop_patience"])),
    ]
    return argv


def parse_best_results(log_path: Path) -> dict[str, float]:
    results: dict[str, float] = {}
    if not log_path.is_file():
        return results
    in_block = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Best Results"):
            in_block = True
            continue
        if in_block and line.startswith("Last Epoch"):
            break
        if in_block:
            m = RESULT_LINE_RE.match(line)
            if not m:
                continue
            key, raw = m.groups()
            val = float(raw)
            if key in ("Acc-2", "Acc-7", "Acc-5", "Acc-3"):
                val /= 100.0
            if key == "Selection score":
                key = "SelectionScore"
            results[key] = val
    return results


def cmd_list_values(axis: str) -> int:
    for v in default_grid(axis):
        print(v)
    return 0


def cmd_queue_plan() -> int:
    for axis in SUPPORTED_AXES:
        for v in default_grid(axis):
            if is_center(axis, v):
                continue
            print(f"{axis}\t{v}")
    return 0


def cmd_train(axis: str, value: float, runs_root: Path, dry_run: bool) -> int:
    if axis not in SUPPORTED_AXES:
        print(f"ERROR: axis must be one of {SUPPORTED_AXES}", file=sys.stderr)
        return 2
    if is_center(axis, value):
        print(f"SKIP center {axis}={value} (use --inject-center on aggregate)", file=sys.stderr)
        return 0
    overrides = {axis: float(value)}
    out_dir = runs_root / axis / value_tag(axis, float(value))
    ckpt = out_dir / "checkpoints"
    argv = build_train_argv(checkpoint_dir=str(ckpt), overrides=overrides)
    cmd = [sys.executable, "-u", str(_TRAIN), *argv]
    if dry_run:
        print("cwd:", _MY)
        print("out:", out_dir)
        print("exec:", " ".join(cmd))
        return 0
    if (out_dir / "train.log").is_file() and parse_best_results(out_dir / "train.log"):
        print(f"SKIP already done {out_dir}")
        return 0
    ckpt.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(
        json.dumps({"axis": axis, "value": float(value), "base": "mosei_deberta_trial36"}) + "\n",
        encoding="utf-8",
    )
    log_path = out_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as logf:
        subprocess.run(cmd, cwd=str(_MY), stdout=logf, stderr=subprocess.STDOUT, check=True)
    return 0


def cmd_aggregate(runs_root: Path, summary_out: Path | None, inject_center: bool) -> int:
    rows: list[dict[str, object]] = []
    for log_path in sorted(runs_root.glob("*/*/train.log")):
        rel = log_path.relative_to(runs_root)
        axis, tag = rel.parts[0], rel.parts[1]
        run_dir = log_path.parent
        meta_path = run_dir / "run_meta.json"
        value_str = ""
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                value_str = str(meta.get("value", ""))
                if meta.get("axis"):
                    axis = str(meta["axis"])
            except (json.JSONDecodeError, OSError):
                pass
        if not value_str and tag.startswith("v"):
            value_str = tag[1:].replace("p", ".").replace("m", "-")
        metrics = parse_best_results(log_path)
        if not metrics:
            continue
        row: dict[str, object] = {
            "axis": axis,
            "value": value_str,
            "tag": tag,
            "log_path": str(log_path),
        }
        row.update({k: metrics.get(k, "") for k in ("MAE", "Corr", "F1", "Acc-2", "Acc-7", "SelectionScore")})
        rows.append(row)

    if inject_center:
        for axis in SUPPORTED_AXES:
            v = float(TRIAL36[axis])
            tag = value_tag(axis, v)
            # drop any incomplete center row
            rows = [r for r in rows if not (r["axis"] == axis and is_center(axis, float(r["value"] or "nan")))]
            row = {
                "axis": axis,
                "value": str(v),
                "tag": tag,
                "log_path": "injected:sota_search/runs/mosei_deberta/trial_36",
            }
            row.update(CENTER_METRICS)
            rows.append(row)

    summary = summary_out.expanduser().resolve() if summary_out else (runs_root / "summary.csv")
    summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "axis", "value", "tag", "MAE", "Corr", "F1", "Acc-2", "Acc-7", "SelectionScore", "log_path",
    ]
    rows.sort(key=lambda r: (str(r["axis"]), float(r["value"]) if str(r["value"]).replace(".", "", 1).replace("-", "", 1).isdigit() or _is_float(r["value"]) else 0.0))
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            for fn in fieldnames:
                r.setdefault(fn, "")
            w.writerow(r)
    print(f"Wrote {summary} ({len(rows)} runs)")
    return 0


def _is_float(x: object) -> bool:
    try:
        float(str(x))
        return True
    except (TypeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-values")
    p_list.add_argument("--axis", required=True, choices=SUPPORTED_AXES)

    sub.add_parser("queue-plan")

    p_train = sub.add_parser("train")
    p_train.add_argument("--axis", required=True, choices=SUPPORTED_AXES)
    p_train.add_argument("--value", type=float, required=True)
    p_train.add_argument(
        "--runs-root",
        type=Path,
        default=_MY / "runs/sensitivity_mosei/trial36",
    )
    p_train.add_argument("--dry-run", action="store_true")

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--runs-root", type=Path, required=True)
    p_agg.add_argument("--summary-out", type=Path, default=None)
    p_agg.add_argument(
        "--inject-center",
        action="store_true",
        help="Inject trial-36 Best Results as the center point of every axis",
    )

    args = ap.parse_args()
    if args.command == "list-values":
        return cmd_list_values(args.axis)
    if args.command == "queue-plan":
        return cmd_queue_plan()
    if args.command == "train":
        return cmd_train(args.axis, args.value, args.runs_root.expanduser().resolve(), args.dry_run)
    if args.command == "aggregate":
        so = args.summary_out.expanduser().resolve() if args.summary_out else None
        return cmd_aggregate(args.runs_root.expanduser().resolve(), so, args.inject_center)
    return 1


if __name__ == "__main__":
    sys.exit(main())
