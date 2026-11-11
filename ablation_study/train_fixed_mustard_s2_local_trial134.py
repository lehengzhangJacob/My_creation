#!/usr/bin/env python3
"""Launch ``ablation_study/train_classify.py`` with frozen MUStARD s2_local trial 134 hyperparameters."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from classify_hparams_common import (
    CONFIDENCE_MODE_CHOICES,
    ROUTING_MODE_CHOICES,
)
from mustard_s2_local_trial134_hparams import PAPER_TEST_METRICS, build_train_argv


def _default_checkpoint_dir(my_creation: Path, ablation: str) -> Path:
    base = my_creation / "ablation_study/runs"
    if ablation == "none":
        return base / "mustard_s2_local_trial134/checkpoints"
    return base / f"mustard_s2_local_trial134_{ablation}/checkpoints"


def main() -> int:
    here = Path(__file__).resolve().parent
    my_creation = here.parent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ablation",
        default="none",
        choices=(
            "none",
            "no_infogate",
            "no_mselector",
            "no_ib",
            "no_ib_no_mselector_no_infogate",
            "no_conf",
            "no_conf_gating",
            "no_adaptive_gate",
        ),
        help=(
            "PRISM ablation: ``no_mselector``=w/o DPR, ``no_ib``=w/o VTB; "
            "``no_conf`` aliases ``no_conf_gating`` in train_classify.py."
        ),
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Passed to train_classify.py --checkpoint_dir "
            "(default: runs/mustard_s2_local_trial134[_<ablation>]/checkpoints)"
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="Print train_classify.py invocation and exit")
    ap.add_argument("--seed", type=int, default=None, help="Override train_classify.py --seed")
    ap.add_argument(
        "--disable-hcf", action="store_true",
        help="Run the three-modality control without handcrafted features.")
    ap.add_argument(
        "--confidence-mode", choices=CONFIDENCE_MODE_CHOICES, default=None,
        help="Override InfoGate confidence granularity.")
    ap.add_argument(
        "--routing-mode", choices=ROUTING_MODE_CHOICES, default=None,
        help="Override primary-modality routing.")
    args = ap.parse_args()

    ckpt = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else _default_checkpoint_dir(my_creation, args.ablation)
    )

    train_py = here / "train_classify.py"
    argv = build_train_argv(
        checkpoint_dir=str(ckpt),
        ablation=args.ablation,
        seed=args.seed,
        disable_hcf=args.disable_hcf,
        confidence_mode=args.confidence_mode,
        routing_mode=args.routing_mode,
    )

    cmd = [sys.executable, "-u", str(train_py), *argv]
    if args.dry_run:
        print("cwd:", my_creation)
        print("train:", train_py)
        print("paper test targets:", PAPER_TEST_METRICS)
        print("exec:", " ".join(cmd))
        return 0

    subprocess.run(cmd, cwd=str(my_creation), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
