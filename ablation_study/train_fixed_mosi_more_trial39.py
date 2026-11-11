#!/usr/bin/env python3
"""Launch ``ablation_study/train.py`` with frozen MOSI more/mosi trial 39 hyperparameters."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mosi_more_trial39_hparams import build_train_argv


def _default_checkpoint_dir(my_creation: Path, ablation: str) -> Path:
    base = my_creation / "ablation_study/runs"
    if ablation == "none":
        return base / "mosi_more_trial39/checkpoints"
    return base / f"mosi_more_trial39_{ablation}/checkpoints"


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
            "``no_conf`` aliases ``no_conf_gating`` in train.py."
        ),
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Passed to train.py --checkpoint_dir (default: runs/mosi_more_trial39[_<ablation>]/checkpoints)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print train.py invocation and exit",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override train.py --seed (default: frozen TRIAL_39_PARAMS['seed']=128)",
    )
    ap.add_argument("--model", default=None, help="Override train.py --model.")
    ap.add_argument(
        "--text-backbone",
        choices=("auto", "deberta", "bert"),
        default=None,
        help="Override train.py --text_backbone.",
    )
    ap.add_argument(
        "--model-variant",
        choices=("prism", "text_only"),
        default=None,
        help="Override train.py --model_variant.",
    )
    ap.add_argument(
        "--confidence-mode",
        choices=("token", "modality", "uniform"),
        default=None,
        help="Override train.py --confidence_mode.",
    )
    ap.add_argument(
        "--routing-mode",
        choices=("gumbel", "argmax", "soft", "batch_gumbel"),
        default=None,
        help="Override train.py --routing_mode.",
    )
    ap.add_argument(
        "--disable-l-rib",
        action="store_true",
        help="Pass train.py --disable_l_rib.",
    )
    ap.add_argument(
        "--step-loss-csv",
        default=None,
        help="Optional path passed to train.py --step_loss_csv.",
    )
    ap.add_argument(
        "--step-loss-log-interval",
        type=int,
        default=1,
        help="Optional value passed to train.py --step_loss_log_interval.",
    )
    args = ap.parse_args()

    ckpt = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else _default_checkpoint_dir(my_creation, args.ablation)
    )

    train_py = here / "train.py"
    argv = build_train_argv(
        checkpoint_dir=str(ckpt),
        ablation=args.ablation,
        seed=args.seed,
        model=args.model,
        text_backbone=args.text_backbone,
        model_variant=args.model_variant,
        confidence_mode=args.confidence_mode,
        routing_mode=args.routing_mode,
        disable_l_rib=args.disable_l_rib,
    )
    if args.step_loss_csv:
        argv.extend(["--step_loss_csv", str(Path(args.step_loss_csv).expanduser().resolve())])
        argv.extend(["--step_loss_log_interval", str(int(args.step_loss_log_interval))])
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
