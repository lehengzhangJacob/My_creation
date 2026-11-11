#!/usr/bin/env python3
"""Replay a frozen Nov-2026 trial (see ``sota_2026_11_trials.py``) and check reproduction.

Examples (from anywhere):
    python fixed_experiment/train_fixed_sota_2026_11.py --trial mustard_trial1239 --dry-run
    python fixed_experiment/train_fixed_sota_2026_11.py --trial mustard_trial1239
    python fixed_experiment/train_fixed_sota_2026_11.py --trial mustard_trial1239 --check-only

After training, the ``Best Results`` block of the new log is compared line by line
with the original trial log; exit code 0 means an exact reproduction.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sota_2026_11_trials import TRIALS, build_argv

HERE = Path(__file__).resolve().parent
MY_CREATION = HERE.parent


def best_block(log_path: Path) -> list[str]:
    """Last ``Best Results`` block (test metrics at the dev-selected checkpoint)."""
    block: list[str] = []
    cur: list[str] | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("Best Results"):
            cur = [line.strip()]
            continue
        if cur is not None:
            if line.startswith("Last Epoch") or not line.strip():
                block, cur = cur, None
            else:
                cur.append(line.strip())
    return cur if cur is not None else block


def check(name: str, log_path: Path) -> bool:
    ref = best_block(MY_CREATION / TRIALS[name]["source"] / "train.log")
    new = best_block(log_path)
    same = bool(new) and new == ref
    print(f"[{name}] original: {TRIALS[name]['test']}")
    for a, b in zip(ref, new + [""] * (len(ref) - len(new))):
        print(f"  {'==' if a == b else '!='}  orig: {a:<40s} new: {b}")
    print(f"[{name}] {'EXACT REPRODUCTION' if same else 'MISMATCH'}")
    return same


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", required=True, choices=sorted(TRIALS))
    ap.add_argument("--out-dir", default=None,
                    help="default: fixed_experiment/runs/sota_2026_11/<trial>")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="only compare an existing run log with the original")
    ap.add_argument("--ablation", default="none",
                    choices=("none", "no_ib", "no_mselector", "no_infogate"),
                    help="no_ib = w/o VTB, no_mselector = w/o DPR, no_infogate = w/o InfoGate; "
                         "all other hyper-parameters stay frozen")
    args = ap.parse_args()

    run_name = args.trial if args.ablation == "none" else f"{args.trial}_{args.ablation}"
    out = Path(args.out_dir or HERE / "runs" / "sota_2026_11" / run_name).resolve()
    log_path = out / "train.log"
    if args.check_only:
        return 0 if check(args.trial, log_path) else 1

    cfg = TRIALS[args.trial]
    cmd = [sys.executable, "-u", cfg["script"], *build_argv(args.trial, str(out / "checkpoints"))]
    if args.ablation != "none":
        cmd += ["--ablation", args.ablation]
    if args.dry_run:
        print("cwd:", MY_CREATION)
        print("exec:", " ".join(cmd))
        print("expected test:", cfg["test"], f"(runtime {cfg['runtime']})")
        return 0

    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[{args.trial}] log: {log_path}", flush=True)
    with open(log_path, "w") as lf:
        rc = subprocess.call(cmd, cwd=str(MY_CREATION), stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"[{args.trial}] training exited with code {rc}; see {log_path}")
        return rc
    if args.ablation != "none":
        print(f"[{run_name}] full model: {cfg['test']}")
        print("\n".join(best_block(log_path)))
        return 0
    return 0 if check(args.trial, log_path) else 1


if __name__ == "__main__":
    sys.exit(main())
