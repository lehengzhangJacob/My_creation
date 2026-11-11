#!/usr/bin/env python3
"""Opportunistic multi-row Optuna search on shared GPUs.

One process drives every row with Optuna's ask/tell API. A trial is launched
only when some GPU has enough free memory for that row; training processes run
in their own session, so restarting the scheduler re-adopts them from
``state.json`` instead of killing them.

    cd My_creation && nohup python sota_search/scheduler.py >> sota_search/runs/scheduler.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rows import MY_CREATION, ROWS, evaluate, sample_params, searchable_gold  # noqa: E402

RUNS = os.path.join(MY_CREATION, "sota_search", "runs")
STATE = os.path.join(RUNS, "state.json")
RESULTS = os.path.join(RUNS, "results.jsonl")
STATUS = os.path.join(RUNS, "STATUS.md")
# The AutoDL ITHP5090 env (Python 3.10, torch 2.9.1) that produced the paper runs.
# The local env of the same name is Python 3.8 and cannot import ablation_study/.
PYTHON = os.environ.get(
    "PYTHON",
    "/home/gmn/danger/codes/PRISM2/_from_seetacloud/autodl-tmp/anaconda3/envs/ITHP5090/bin/python")

RESULT_RE = re.compile(r"^\s+([A-Za-z0-9\- ]+?):\s+([-+]?[0-9]*\.?[0-9]+)(%?)\s*$")
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED")
KEY_MAP = {"Acc-2": "Acc2", "Acc-7": "Acc7", "Acc-5": "Acc5", "Acc-3": "Acc3",
           "MAE": "MAE", "Corr": "Corr", "F1": "F1", "Acc": "Acc"}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def load_state() -> dict:
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"running": [], "done_rows": [], "mem": {}, "best": {}, "oom_retries": {}}


def save_state(state: dict) -> None:
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE)


def gpu_free() -> dict[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout
    free = {}
    for line in out.strip().splitlines():
        idx, mib = (x.strip() for x in line.split(","))
        free[int(idx)] = int(mib)
    return free


def pid_mem() -> dict[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout
    used = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            used[int(parts[0])] = used.get(int(parts[0]), 0) + int(parts[1])
    return used


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return False


def parse_best(log_path: str) -> dict:
    """Test metrics at the dev-selected checkpoint (``Best Results`` block)."""
    res = {}
    if not os.path.exists(log_path):
        return res
    block = False
    with open(log_path, errors="replace") as f:
        for line in f:
            if line.startswith("Best Results"):
                block, res = True, {}
                continue
            if block and (line.startswith("Last Epoch") or not line.strip()):
                block = False
                continue
            if block:
                m = RESULT_RE.match(line.rstrip("\n"))
                if not m:
                    continue
                key, val, pct = m.group(1).strip(), float(m.group(2)), m.group(3)
                if key not in KEY_MAP:
                    continue
                k = KEY_MAP[key]
                if k == "F1" and not pct and val <= 1.0:
                    val *= 100.0
                res[k] = val
    return res


def log_has_oom(log_path: str) -> bool:
    try:
        with open(log_path, errors="replace") as f:
            tail = f.read()[-20000:]
    except OSError:
        return False
    return any(m in tail for m in OOM_MARKERS)


def study_for(row) -> optuna.Study:
    d = os.path.join(RUNS, row.name)
    os.makedirs(d, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=12, multivariate=True, constant_liar=True, seed=None)
    study = optuna.create_study(
        study_name=row.name, storage=f"sqlite:///{os.path.join(d, 'study.db')}",
        direction="maximize", sampler=sampler, load_if_exists=True)
    if len(study.trials) == 0:
        study.enqueue_trial(searchable_gold(row), user_attrs={"kind": "gold"})
        for s in (1, 7, 42, 2024, 3407, 9999)[: row.seed_probes]:
            study.enqueue_trial(searchable_gold(row, seed=s), user_attrs={"kind": "seed_probe"})
    return study


def maybe_enqueue_exploit(study, exploit: float, top_k: int) -> None:
    if random.random() >= exploit:
        return
    if any(t.state == optuna.trial.TrialState.WAITING for t in study.trials):
        return
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if len(done) < 12:
        return
    base = random.choice(sorted(done, key=lambda t: -t.value)[:top_k])
    params = dict(base.params, seed=random.randint(0, 9999))
    study.enqueue_trial(params, user_attrs={"kind": "exploit", "base": base.number})


def launch(row, study, gpu: int, state: dict, exploit: float = 0.0, top_k: int = 3) -> subprocess.Popen:
    maybe_enqueue_exploit(study, exploit, top_k)
    trial = study.ask()
    kind = trial.user_attrs.get("kind", "tpe")
    params = sample_params(trial, row)
    if kind == "gold":
        params.pop("early_stop_patience", None)
        if "early_stop_patience" in row.gold:
            params["early_stop_patience"] = row.gold["early_stop_patience"]
    tdir = os.path.join(RUNS, row.name, f"trial_{trial.number}")
    os.makedirs(tdir, exist_ok=True)
    ckpt = os.path.join(tdir, "checkpoints")
    log_path = os.path.join(tdir, "train.log")
    argv = row.build_argv(params, ckpt)
    cmd = [PYTHON, "-u", row.script, *argv]
    with open(os.path.join(tdir, "cmd.json"), "w") as f:
        json.dump({"cmd": cmd, "params": params, "gpu": gpu, "kind": kind}, f, indent=1)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(k, None)
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=MY_CREATION, stdout=lf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    state["running"].append({
        "row": row.name, "trial": trial.number, "pid": proc.pid, "gpu": gpu,
        "log": log_path, "ckpt": ckpt, "params": params, "kind": kind,
        "started": now(), "peak_mib": 0,
    })
    log(f"launch {row.name} trial {trial.number} ({kind}) on GPU{gpu} pid={proc.pid}")
    return proc


def finish(job: dict, study, state: dict, procs: dict) -> None:
    row = ROWS[job["row"]]
    proc = procs.pop(job["pid"], None)
    rc = proc.returncode if proc is not None else None
    res = parse_best(job["log"])
    oom = log_has_oom(job["log"])
    if oom and not res:
        key = f"{row.name}:{job['trial']}"
        retries = state["oom_retries"].get(key, 0)
        study.tell(job["trial"], state=optuna.trial.TrialState.FAIL)
        state["mem"][row.name] = max(state["mem"].get(row.name, row.mem_mib),
                                     int(job.get("peak_mib", 0) * 1.15))
        if retries < 3:
            t = study.trials[job["trial"]]
            study.enqueue_trial(t.params, user_attrs={"kind": job["kind"], "retry_of": job["trial"]})
            state["oom_retries"][f"{row.name}:{job['trial']}"] = retries + 1
        shutil.rmtree(job["ckpt"], ignore_errors=True)
        log(f"OOM {row.name} trial {job['trial']} -> requeued, mem est {state['mem'][row.name]} MiB")
        return
    if not res:
        study.tell(job["trial"], state=optuna.trial.TrialState.FAIL)
        shutil.rmtree(job["ckpt"], ignore_errors=True)
        log(f"FAIL {row.name} trial {job['trial']} rc={rc} (no Best Results)")
        return
    score, met, margins = evaluate(row, res)
    study.tell(job["trial"], score)
    record = {"time": now(), "row": row.name, "trial": job["trial"], "kind": job["kind"],
              "score": score, "met": met, "results": res, "margins": margins,
              "params": job["params"], "gpu": job["gpu"], "peak_mib": job.get("peak_mib", 0),
              "log": job["log"]}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")
    best = state["best"].get(row.name)
    if best is None or (met, score) > (bool(best.get("met")), best["score"]):
        # The previous best is kept only if it met the targets (it may already be
        # referenced as a paper result); older met checkpoints stay on disk.
        if best and not best.get("met") and best.get("ckpt") and best["ckpt"] != job["ckpt"]:
            shutil.rmtree(best["ckpt"], ignore_errors=True)
        state["best"][row.name] = {"score": score, "met": met, "trial": job["trial"],
                                   "results": res, "ckpt": job["ckpt"]}
        log(f"new best {row.name} trial {job['trial']} score={score:.3f} met={met}")
    else:
        shutil.rmtree(job["ckpt"], ignore_errors=True)
    if met and row.name not in state["done_rows"]:
        state["done_rows"].append(row.name)
        log(f"*** TARGETS MET {row.name} trial {job['trial']}: {res}")
    log(f"done {row.name} trial {job['trial']} score={score:.3f} met={met} {res}")


def write_status(state: dict, studies: dict) -> None:
    lines = [f"# SOTA search status ({now()})", ""]
    lines.append("| row | met | best trial | best score | best metrics | complete | running |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, row in ROWS.items():
        st = studies[name]
        n_done = sum(1 for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE)
        n_run = sum(1 for j in state["running"] if j["row"] == name)
        b = state["best"].get(name)
        if b:
            mets = " ".join(f"{m.key}={b['results'].get(m.key, float('nan')):.4g}"
                            f"(≥{m.target})" if m.higher else
                            f"{m.key}={b['results'].get(m.key, float('nan')):.4g}(≤{m.target})"
                            for m in row.metrics)
            lines.append(f"| {name} | {'YES' if b['met'] else 'no'} | {b['trial']} | "
                         f"{b['score']:.3f} | {mets} | {n_done} | {n_run} |")
        else:
            lines.append(f"| {name} | no | - | - | - | {n_done} | {n_run} |")
    lines += ["", "## running", ""]
    for j in state["running"]:
        lines.append(f"- {j['row']} trial {j['trial']} ({j['kind']}) GPU{j['gpu']} "
                     f"pid {j['pid']} since {j['started']} peak {j.get('peak_mib', 0)} MiB")
    with open(STATUS, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=",".join(ROWS))
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--max_per_gpu", type=int, default=2)
    ap.add_argument("--max_per_row", type=int, default=2)
    ap.add_argument("--margin_mib", type=int, default=800)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--launch_cooldown", type=int, default=150,
                    help="seconds to wait after a launch on a GPU before trusting its free memory")
    ap.add_argument("--starve_after", type=int, default=1200,
                    help="seconds a row may sit with nothing running before a GPU is reserved for it")
    ap.add_argument("--python", default=None)
    ap.add_argument("--exploit", type=float, default=0.5,
                    help="probability of re-running a top config with a fresh seed")
    ap.add_argument("--exploit_top_k", type=int, default=3)
    ap.add_argument("--pause_rows", default="",
                    help="comma-separated rows that get no new trials until --pause_until")
    ap.add_argument("--pause_until", default="",
                    help="'YYYY-mm-dd HH:MM:SS'; paused rows resume after this time")
    ap.add_argument("--no_keep_going", dest="keep_going", action="store_false",
                    help="stop searching a row once its targets are met")
    args = ap.parse_args()
    if args.python:
        global PYTHON
        PYTHON = args.python

    os.makedirs(RUNS, exist_ok=True)
    active_rows = [r for r in args.rows.split(",") if r in ROWS]
    gpus = [int(g) for g in args.gpus.split(",")]
    state = load_state()
    studies = {name: study_for(ROWS[name]) for name in ROWS}
    procs: dict[int, subprocess.Popen] = {}
    last_launch: dict[int, float] = {}
    row_last_launch: dict[str, float] = {j["row"]: time.time() for j in state["running"]}
    reserved_log: dict = {}
    paused_log: dict = {}

    tracked = {(j["row"], j["trial"]) for j in state["running"]}
    for name, st in studies.items():
        for t in st.trials:
            if t.state == optuna.trial.TrialState.RUNNING and (name, t.number) not in tracked:
                st.tell(t.number, state=optuna.trial.TrialState.FAIL)
                if t.params:
                    st.enqueue_trial(t.params, user_attrs={
                        "kind": t.user_attrs.get("kind", "tpe"), "retry_of": t.number})
                log(f"stale RUNNING trial {name}#{t.number} marked FAIL and requeued")
    log(f"scheduler start rows={active_rows} gpus={gpus} adopted={len(state['running'])}")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    while not stop["flag"]:
        used = pid_mem()
        for j in list(state["running"]):
            j["peak_mib"] = max(j.get("peak_mib", 0), used.get(j["pid"], 0))
            p = procs.get(j["pid"])
            running = p.poll() is None if p is not None else alive(j["pid"])
            if not running:
                state["running"].remove(j)
                finish(j, studies[j["row"]], state, procs)
                row = ROWS[j["row"]]
                if j.get("peak_mib"):
                    prev = state["mem"].get(row.name, row.mem_mib)
                    state["mem"][row.name] = int(max(j["peak_mib"] * 1.1, min(prev, j["peak_mib"] * 1.5)))
        save_state(state)

        free = gpu_free()
        paused = set(args.pause_rows.split(",")) if args.pause_rows and now() < args.pause_until else set()
        if paused != paused_log.get("rows"):
            log(f"paused rows {sorted(paused)} until {args.pause_until}" if paused else "pause over")
            paused_log["rows"] = paused
        live_rows = [r for r in active_rows if r not in paused]
        unmet = [r for r in live_rows if r not in state["done_rows"]]
        pending = list(live_rows) if args.keep_going else unmet
        starving = [r for r in unmet
                    if not any(j["row"] == r for j in state["running"])
                    and time.time() - row_last_launch.get(r, 0) > args.starve_after
                    and all(state["mem"].get(r, ROWS[r].mem_mib) > free.get(g, 0) - args.margin_mib
                            for g in gpus)]
        reserved = None
        if starving:
            reserved = max(gpus, key=lambda g: free.get(g, 0))
            if reserved_log.get("gpu") != reserved or reserved_log.get("rows") != starving:
                log(f"reserving GPU{reserved} for starving rows {starving}")
                reserved_log.update(gpu=reserved, rows=starving)
        for gpu in gpus:
            if time.time() - last_launch.get(gpu, 0) < args.launch_cooldown:
                continue
            ours = [j for j in state["running"] if j["gpu"] == gpu]
            if len(ours) >= args.max_per_gpu:
                continue
            avail = free.get(gpu, 0) - args.margin_mib
            cands = []
            for name in pending:
                if gpu == reserved and name not in starving:
                    continue
                n_run = sum(1 for j in state["running"] if j["row"] == name)
                if n_run >= args.max_per_row:
                    continue
                need = state["mem"].get(name, ROWS[name].mem_mib)
                if need <= avail:
                    n_done = sum(1 for t in studies[name].trials
                                 if t.state == optuna.trial.TrialState.COMPLETE)
                    cands.append((name not in unmet, n_run, n_done, name))
            if not cands:
                continue
            cands.sort()
            name = cands[0][-1]
            proc = launch(ROWS[name], studies[name], gpu, state, args.exploit, args.exploit_top_k)
            procs[proc.pid] = proc
            last_launch[gpu] = time.time()
            row_last_launch[name] = time.time()
            save_state(state)
            free[gpu] = free.get(gpu, 0) - state["mem"].get(name, ROWS[name].mem_mib)
        write_status(state, studies)
        if not pending and not state["running"]:
            log("all rows met their targets; exiting")
            break
        time.sleep(args.poll)
    save_state(state)
    log("scheduler exit")


if __name__ == "__main__":
    main()
