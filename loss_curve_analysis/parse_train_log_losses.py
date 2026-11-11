#!/usr/bin/env python3
"""Parse MOSI/MOSEI training logs into paper-style weighted loss CSVs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


MY_CREATION = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

EPOCH_RE = re.compile(r"^Epoch\s+(\d+)/(\d+)\s+\[stage\s+(\d+)\]")
LOSS_RE = re.compile(
    r"Loss\s+total=([-+0-9.eE]+)\s+task=([-+0-9.eE]+)\s+ib=([-+0-9.eE]+)"
)
DETAIL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([-+0-9.eE]+)")


@dataclass(frozen=True)
class CaseConfig:
    dataset: str
    trial: str
    log_path: Path
    hparams_path: Path
    params_name: str
    gradient_accumulation_name: str
    output_name: str


CASES = (
    CaseConfig(
        dataset="mosi",
        trial="more_mosi_trial39",
        log_path=MY_CREATION
        / "logs/optuna/4090D_restart/more/mosi/train_logs/mosi_more_mosi_trial_39.log",
        hparams_path=MY_CREATION / "ablation_study/mosi_more_trial39_hparams.py",
        params_name="TRIAL_39_PARAMS",
        gradient_accumulation_name="GRADIENT_ACCUMULATION_STEP",
        output_name="mosi_trial39_losses.csv",
    ),
    CaseConfig(
        dataset="mosei",
        trial="phase1_trial70",
        log_path=MY_CREATION / "fixed_experiment/runs/mosei_phase1_trial70/train.log",
        hparams_path=MY_CREATION / "fixed_experiment/mosei_phase1_trial70_hparams.py",
        params_name="TRIAL_70_PARAMS",
        gradient_accumulation_name="GRADIENT_ACCUMULATION_STEP",
        output_name="mosei_trial70_losses.csv",
    ),
)

FIELDNAMES = [
    "dataset",
    "trial",
    "epoch",
    "total_epochs",
    "stage",
    "stage1_epochs",
    "alpha_ib",
    "selector_rib_weight",
    "gradient_accumulation_step",
    "logged_loss_total_scaled",
    "logged_loss_total_unscaled",
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
    "L_total",
    "L_total_task_plus_ib",
    "parts_minus_task_plus_ib",
    "unscaled_minus_task_plus_ib",
]


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_detail(line: str) -> dict[str, float]:
    return {key: float(value) for key, value in DETAIL_RE.findall(line)}


def _case_weights(case: CaseConfig) -> tuple[float, float, int, int]:
    module = _load_module(case.hparams_path, f"_loss_curve_{case.dataset}_{case.trial}")
    params = getattr(module, case.params_name)
    alpha_ib = float(params["alpha_ib"])
    selector_rib_weight = float(params["selector_rib_weight"])
    stage1_epochs = int(params["stage1_epochs"])
    gradient_accumulation_step = int(getattr(module, case.gradient_accumulation_name))
    return alpha_ib, selector_rib_weight, stage1_epochs, gradient_accumulation_step


def parse_case(case: CaseConfig) -> list[dict[str, float | int | str]]:
    if not case.log_path.exists():
        raise FileNotFoundError(case.log_path)
    if not case.hparams_path.exists():
        raise FileNotFoundError(case.hparams_path)

    alpha_ib, selector_rib_weight, stage1_epochs, grad_accum = _case_weights(case)
    rows: list[dict[str, float | int | str]] = []
    current: dict[str, float | int | str] | None = None

    with case.log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            epoch_match = EPOCH_RE.match(line)
            if epoch_match:
                current = {
                    "dataset": case.dataset,
                    "trial": case.trial,
                    "epoch": int(epoch_match.group(1)),
                    "total_epochs": int(epoch_match.group(2)),
                    "stage": int(epoch_match.group(3)),
                    "stage1_epochs": stage1_epochs,
                    "alpha_ib": alpha_ib,
                    "selector_rib_weight": selector_rib_weight,
                    "gradient_accumulation_step": grad_accum,
                }
                continue

            if current is None:
                continue

            loss_match = LOSS_RE.search(line)
            if loss_match:
                logged_scaled = float(loss_match.group(1))
                current["logged_loss_total_scaled"] = logged_scaled
                current["logged_loss_total_unscaled"] = logged_scaled * grad_accum
                current["L_task"] = float(loss_match.group(2))
                current["ib_logged"] = float(loss_match.group(3))
                continue

            if line.startswith("Detail"):
                detail = _parse_detail(line)
                l_tib = detail.get("L_tib", 0.0)
                l_lib = detail.get("L_lib", 0.0)
                l_rib = detail.get("L_rib", 0.0)
                l_tib_weighted = alpha_ib * l_tib
                l_lib_weighted = alpha_ib * l_lib
                l_rib_weighted = selector_rib_weight * l_rib
                task = float(current["L_task"])
                ib_logged = float(current["ib_logged"])
                l_total = task + l_tib_weighted + l_lib_weighted + l_rib_weighted
                task_plus_ib = task + ib_logged

                current.update(
                    {
                        "L_tib_raw": l_tib,
                        "L_lib_raw": l_lib,
                        "L_rib_raw": l_rib,
                        "L_rib_kl_raw": detail.get("L_rib_kl", 0.0),
                        "L_rib_balance_raw": detail.get("L_rib_balance", 0.0),
                        "L_tib_weighted": l_tib_weighted,
                        "L_lib_weighted": l_lib_weighted,
                        "L_rib_weighted": l_rib_weighted,
                        "L_total": l_total,
                        "L_total_task_plus_ib": task_plus_ib,
                        "parts_minus_task_plus_ib": l_total - task_plus_ib,
                        "unscaled_minus_task_plus_ib": float(
                            current["logged_loss_total_unscaled"]
                        )
                        - task_plus_ib,
                    }
                )
                rows.append(current)
                current = None

    if not rows:
        raise RuntimeError(f"No epoch loss rows parsed from {case.log_path}")
    return rows


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    max_parts_diff = max(abs(float(row["parts_minus_task_plus_ib"])) for row in rows)
    max_unscaled_diff = max(abs(float(row["unscaled_minus_task_plus_ib"])) for row in rows)
    stage1_rib_values = [
        abs(float(row["L_rib_weighted"]))
        for row in rows
        if int(row["epoch"]) <= int(row["stage1_epochs"])
    ]
    max_stage1_rib = max(stage1_rib_values, default=0.0)
    return {
        "rows": len(rows),
        "max_abs_parts_minus_task_plus_ib": max_parts_diff,
        "max_abs_unscaled_minus_task_plus_ib": max_unscaled_diff,
        "max_abs_stage1_l_rib_weighted": max_stage1_rib,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for parsed CSV files.",
    )
    args = parser.parse_args()

    for case in CASES:
        rows = parse_case(case)
        output_path = args.output_dir / case.output_name
        write_csv(rows, output_path)
        validation = validate_rows(rows)
        print(f"Wrote {output_path}")
        print(
            "  rows={rows} max|parts-(task+ib)|={parts:.6g} "
            "max|scaled_total*grad-(task+ib)|={scaled:.6g} "
            "max stage1 L_rib_w={rib:.6g}".format(
                rows=validation["rows"],
                parts=validation["max_abs_parts_minus_task_plus_ib"],
                scaled=validation["max_abs_unscaled_minus_task_plus_ib"],
                rib=validation["max_abs_stage1_l_rib_weighted"],
            )
        )


if __name__ == "__main__":
    main()
