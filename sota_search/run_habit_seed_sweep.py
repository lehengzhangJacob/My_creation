#!/usr/bin/env python3
"""Re-run frozen paper/SOTA trial hparams under conventional habit seeds.

Does not go through Optuna ask() (avoids polluting TPE). Writes under
``runs/<row>/habit_seed/base<trial>_s<seed>/`` and appends
``runs/habit_seed_results.jsonl``.

Usage (from My_creation)::

    python sota_search/run_habit_seed_sweep.py plan \\
        --specs mosi_deberta:812,mosei_deberta:36 --seeds 0,1,2,42,123

    CUDA_VISIBLE_DEVICES=2 python sota_search/run_habit_seed_sweep.py run-one \\
        --row mosi_deberta --base-trial 812 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rows import MY_CREATION, ROWS, evaluate  # noqa: E402
import scheduler as S  # noqa: E402

DEFAULT_SEEDS = (0, 1, 2, 42, 123)
# Priority: DeBERTa main-table winners first, then SIMS; Mustard/BERT optional.
DEFAULT_SPECS = (
    "mosi_deberta:812",
    "mosei_deberta:36",
    "simsv2:159",
)


def _runs_root() -> Path:
    return Path(S.RUNS)


def _habit_results_path() -> Path:
    return _runs_root() / "habit_seed_results.jsonl"


def _load_base_params(row: str, base_trial: int) -> dict:
    cmd = _runs_root() / row / f"trial_{base_trial}" / "cmd.json"
    if not cmd.is_file():
        raise FileNotFoundError(cmd)
    data = json.loads(cmd.read_text())
    params = dict(data["params"])
    return params


def _outdir(row: str, base_trial: int, seed: int) -> Path:
    return _runs_root() / row / "habit_seed" / f"base{base_trial}_s{seed}"


def _already_done(out: Path) -> bool:
    log = out / "train.log"
    if not log.is_file():
        return False
    return S.parse_best(str(log)) is not None


def parse_specs(spec_str: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for part in spec_str.split(","):
        part = part.strip()
        if not part:
            continue
        row, trial_s = part.split(":")
        row = row.strip()
        if row not in ROWS:
            raise SystemExit(f"unknown row {row}")
        out.append((row, int(trial_s)))
    return out


def cmd_plan(specs: list[tuple[str, int]], seeds: list[int]) -> int:
    jobs = []
    for row, base in specs:
        params = _load_base_params(row, base)
        for seed in seeds:
            out = _outdir(row, base, seed)
            status = "done" if _already_done(out) else "need"
            # skip if this seed equals the original winning seed (already known)
            if int(params.get("seed", -1)) == seed and status == "need":
                # still run into habit_seed dir for a clean 5-seed table, unless done
                pass
            jobs.append((row, base, seed, status, ROWS[row].mem_mib))
            print(f"{row}\t{base}\t{seed}\t{status}\t{ROWS[row].mem_mib}")
    need = sum(1 for j in jobs if j[3] == "need")
    print(f"# total={len(jobs)} need={need}", file=sys.stderr)
    return 0


def cmd_run_one(row: str, base_trial: int, seed: int, python: str) -> int:
    rdef = ROWS[row]
    params = _load_base_params(row, base_trial)
    params["seed"] = int(seed)
    out = _outdir(row, base_trial, seed)
    out.mkdir(parents=True, exist_ok=True)
    if _already_done(out):
        print(f"SKIP done {out}")
        return 0
    ckpt = out / "checkpoints"
    log_path = out / "train.log"
    argv = rdef.build_argv(params, str(ckpt))
    cmd = [python, "-u", rdef.script, *argv]
    meta = {
        "row": row,
        "base_trial": base_trial,
        "seed": seed,
        "params": params,
        "cmd": cmd,
        "kind": "habit_seed",
    }
    (out / "cmd.json").write_text(json.dumps(meta, indent=1))
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(k, None)
    print(f"RUN {row} base{base_trial} seed={seed} -> {out}", flush=True)
    with log_path.open("w") as lf:
        rc = subprocess.call(cmd, cwd=MY_CREATION, stdout=lf, stderr=subprocess.STDOUT, env=env)
    res = S.parse_best(str(log_path))
    if not res:
        print(f"FAIL {row} base{base_trial} seed={seed} rc={rc}", flush=True)
        return 1
    score, met, margins = evaluate(rdef, res)
    record = {
        "time": S.now(),
        "row": row,
        "base_trial": base_trial,
        "seed": seed,
        "kind": "habit_seed",
        "score": score,
        "met": met,
        "results": res,
        "margins": margins,
        "params": params,
        "log": str(log_path),
        "rc": rc,
    }
    with _habit_results_path().open("a") as f:
        f.write(json.dumps(record) + "\n")
    tag = "*** MET " if met else ""
    print(f"{tag}DONE {row} base{base_trial} seed={seed} score={score:.3f} met={met} {res}", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument(
        "--specs",
        default=",".join(DEFAULT_SPECS),
        help="Comma list of row:base_trial (default: DeBERTa winners + SIMS).",
    )
    p_plan.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Habit seeds (default: 0,1,2,42,123).",
    )

    p_run = sub.add_parser("run-one")
    p_run.add_argument("--row", required=True)
    p_run.add_argument("--base-trial", type=int, required=True)
    p_run.add_argument("--seed", type=int, required=True)
    p_run.add_argument("--python", default="/home/gmn/.conda/envs/prism/bin/python")

    args = ap.parse_args()
    if args.cmd == "plan":
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        return cmd_plan(parse_specs(args.specs), seeds)
    if args.cmd == "run-one":
        return cmd_run_one(args.row, args.base_trial, args.seed, args.python)
    raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
