#!/usr/bin/env python3
"""Characterize samples whose hard primary is not language (Reviewer 2.1).

Reads the existing DPR dump CSVs. No model reload.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MY = HERE.parent

DEFAULT_INPUTS = (
    ("mosei", MY / "Qualitative_Evaluation/results/mosei_trial70_dpr_test.csv"),
    ("simsv2", MY / "Qualitative_Evaluation/results/simsv2_trial53_dpr_test.csv"),
)


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_groups(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["primary_name"]].append(row)
    summary = []
    for name in ("language", "acoustic", "visual"):
        group = groups.get(name, [])
        labels = [_float(row, "label") for row in group]
        errors = [_float(row, "abs_err") for row in group]
        conf_l = [_float(row, "ib_conf_t") for row in group]
        conf_a = [_float(row, "ib_conf_a") for row in group]
        conf_v = [_float(row, "ib_conf_v") for row in group]
        summary.append({
            "group": name,
            "n": str(len(group)),
            "share": f"{len(group) / len(rows):.4f}" if rows else "nan",
            "label_mean": f"{_mean(labels):.4f}",
            "label_abs_mean": f"{_mean([abs(v) for v in labels]):.4f}",
            "mae": f"{_mean(errors):.4f}",
            "ib_conf_t": f"{_mean(conf_l):.4f}",
            "ib_conf_a": f"{_mean(conf_a):.4f}",
            "ib_conf_v": f"{_mean(conf_v):.4f}",
        })
    return summary


def uncertain_subset(rows: list[dict[str, str]], quantile: float = 0.20) -> dict[str, str]:
    """Low w_max means the router is not committed to one modality."""
    scores = [_float(row, "w_max") for row in rows]
    cutoff = _quantile(scores, quantile)
    chosen = [row for row in rows if _float(row, "w_max") <= cutoff]
    rest = [row for row in rows if _float(row, "w_max") > cutoff]
    return {
        "quantile": f"{quantile:.2f}",
        "w_max_cutoff": f"{cutoff:.4f}",
        "n_uncertain": str(len(chosen)),
        "mae_uncertain": f"{_mean([_float(r, 'abs_err') for r in chosen]):.4f}",
        "mae_rest": f"{_mean([_float(r, 'abs_err') for r in rest]):.4f}",
        "nonlanguage_share_uncertain": (
            f"{sum(r['primary_name'] != 'language' for r in chosen) / len(chosen):.4f}"
            if chosen else "nan"
        ),
    }


def markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines) + "\n"


def analyze(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return summarize_groups(rows), uncertain_subset(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = ["# Non-language primary and uncertain-routing subsets\n"]
    for dataset, path in DEFAULT_INPUTS:
        if not path.is_file():
            print(f"Missing {path}")
            continue
        groups, uncertain = analyze(path)
        group_path = args.out_dir / f"nonlanguage_primary_{dataset}.csv"
        with group_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(groups[0].keys()))
            writer.writeheader()
            writer.writerows(groups)
        uncertain_path = args.out_dir / f"uncertain_routing_{dataset}.csv"
        with uncertain_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(uncertain.keys()))
            writer.writeheader()
            writer.writerow(uncertain)
        report.append(f"## {dataset}\n\n" + markdown_table(groups))
        report.append("Uncertain routing (lowest 20% w_max):\n\n" + markdown_table([uncertain]) + "\n")
        print(f"Wrote {group_path}")
    (args.out_dir / "nonlanguage_primary.md").write_text("".join(report), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'nonlanguage_primary.md'}")


if __name__ == "__main__":
    main()
