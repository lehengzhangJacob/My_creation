#!/usr/bin/env python3
"""Launch text-only DeBERTa on CMU-MOSEI with the clean baseline recipe."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from text_only_deberta_hparams import build_train_argv


def main() -> int:
    here = Path(__file__).resolve().parent
    my_creation = here.parent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Passed to train.py --checkpoint_dir",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    ckpt = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else my_creation / "ablation_study/runs/text_only_deberta_mosei/checkpoints"
    )

    train_py = here / "train.py"
    argv = build_train_argv(
        dataset="mosei",
        checkpoint_dir=str(ckpt),
        seed=args.seed,
        model=args.model,
    )
    cmd = [sys.executable, "-u", str(train_py), *argv]
    if args.dry_run:
        print("cwd:", my_creation)
        print("train:", train_py)
        print("exec:", " ".join(cmd))
        return 0

    subprocess.run(cmd, cwd=str(my_creation), check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
