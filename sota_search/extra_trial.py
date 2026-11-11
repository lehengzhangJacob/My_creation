"""Run one extra trial on a given GPU alongside the running scheduler.

Usage (from My_creation):
    python sota_search/extra_trial.py --row mustard --gpu 1 [--exploit]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scheduler as S  # noqa: E402
from rows import MY_CREATION, ROWS, evaluate, sample_params  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, choices=list(ROWS))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--exploit", action="store_true",
                    help="re-run one of the top-3 configs with a fresh seed")
    ap.add_argument("--python", default="/home/gmn/.conda/envs/prism/bin/python")
    args = ap.parse_args()

    row = ROWS[args.row]
    study = S.study_for(row)
    if args.exploit:
        S.maybe_enqueue_exploit(study, 1.0, 3)
    trial = study.ask()
    kind = "extra_" + trial.user_attrs.get("kind", "tpe")
    params = sample_params(trial, row)
    tdir = os.path.join(S.RUNS, row.name, f"trial_{trial.number}")
    os.makedirs(tdir, exist_ok=True)
    ckpt = os.path.join(tdir, "checkpoints")
    log_path = os.path.join(tdir, "train.log")
    cmd = [args.python, "-u", row.script, *row.build_argv(params, ckpt)]
    with open(os.path.join(tdir, "cmd.json"), "w") as f:
        json.dump({"cmd": cmd, "params": params, "gpu": args.gpu, "kind": kind}, f, indent=1)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false")
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(k, None)
    S.log(f"extra launch {row.name} trial {trial.number} ({kind}) on GPU{args.gpu}")
    with open(log_path, "w") as lf:
        rc = subprocess.call(cmd, cwd=MY_CREATION, stdout=lf, stderr=subprocess.STDOUT, env=env)

    res = S.parse_best(log_path)
    if not res:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        S.log(f"extra FAIL {row.name} trial {trial.number} rc={rc}")
        return
    score, met, margins = evaluate(row, res)
    study.tell(trial.number, score)
    with open(S.RESULTS, "a") as f:
        f.write(json.dumps({"time": S.now(), "row": row.name, "trial": trial.number, "kind": kind,
                            "score": score, "met": met, "results": res, "margins": margins,
                            "params": params, "gpu": args.gpu, "log": log_path}) + "\n")
    tag = "*** TARGETS MET " if met else ""
    S.log(f"{tag}extra done {row.name} trial {trial.number} score={score:.3f} met={met} {res}")


if __name__ == "__main__":
    main()
