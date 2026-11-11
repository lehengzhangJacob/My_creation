#!/usr/bin/env python3
"""Launch MulT / MISA / Self-MM and export per-sample test predictions.

Install first, in a separate env so ITHP5090 stays untouched::

    conda create -n mmsa python=3.10 -y
    conda run -n mmsa pip install MMSA

MMSA's public runner does not write per-sample predictions. This driver
registers a test-loop callback if ``MMSA.trains`` exposes the trainer, and
otherwise tells you the exact function to wrap. MOSEI row order must be
checked against our test labels before the subset comparison.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = {
    "simsv2": HERE.parent / "datasets" / "simsv2.pkl",
    "mosei": HERE.parent / "datasets" / "mosei.pkl",
}
MODELS = {
    "mult": "MulT",
    "misa": "MISA",
    "self_mm": "Self_MM",
}
MMSA_DATASET = {"simsv2": "SIMS", "mosei": "MOSEI"}


def _flatten(values):
    flat = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.append(float(value[0]) if len(value) == 1 else float(value))
        else:
            flat.append(float(value))
    return flat


def export_rows(path: Path, sample_ids, labels, predictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = _flatten(labels)
    predictions = _flatten(predictions)
    if sample_ids is None:
        sample_ids = list(range(len(predictions)))
    if not (len(sample_ids) == len(labels) == len(predictions)):
        raise ValueError(
            f"export length mismatch ids={len(sample_ids)} "
            f"labels={len(labels)} preds={len(predictions)}"
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_index", "true_label", "prediction")
        )
        writer.writeheader()
        for sample_id, label, pred in zip(sample_ids, labels, predictions):
            writer.writerow({
                "sample_index": sample_id,
                "true_label": label,
                "prediction": pred,
            })
    print(f"Wrote {path} ({len(predictions)} rows)")


def _install_export_hook(out_path: Path) -> None:
    """Ask MMSA's test loop for per-sample results and write them to CSV.

    MMSA 2.2 trainers already support ``return_sample_results``. The public
    ``MMSA_run`` path does not turn it on, so this wraps ``ATIO.getTrain``.
    """
    try:
        from MMSA.trains.ATIO import ATIO
    except ImportError as exc:
        raise SystemExit(
            "MMSA is not importable. Use conda env mmsa: "
            "`conda activate mmsa` then `pip install MMSA`."
        ) from exc

    original = ATIO.getTrain

    def get_train(self, args):
        trainer = original(self, args)
        raw = trainer.do_test

        def do_test(model, dataloader, mode="VAL", return_sample_results=False):
            want_rows = return_sample_results or str(mode).upper() == "TEST"
            result = raw(
                model, dataloader, mode=mode, return_sample_results=want_rows
            )
            if (
                str(mode).upper() == "TEST"
                and isinstance(result, dict)
                and "SResults" in result
            ):
                export_rows(
                    out_path,
                    result.get("Ids"),
                    result.get("Labels"),
                    result["SResults"],
                )
            return result

        trainer.do_test = do_test
        return trainer

    ATIO.getTrain = get_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATA), required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--seeds", default="128")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results" / "mmsa")
    args = parser.parse_args()
    feature = DATA[args.dataset]
    if not feature.is_file():
        raise SystemExit(f"missing features: {feature}")
    seeds = [int(item) for item in args.seeds.split(",") if item]
    for seed in seeds:
        out = args.out_dir / f"{args.dataset}_{args.model}_s{seed}.csv"
        _install_export_hook(out)
        from MMSA import MMSA_run
        MMSA_run(
            MODELS[args.model],
            MMSA_DATASET[args.dataset],
            seeds=[seed],
            custom_feature=str(feature),
            gpu_ids=[args.gpu_id],
            num_workers=0,
            model_save_dir=str(args.out_dir / "saved_models"),
            res_save_dir=str(args.out_dir / "mmsa_results"),
            log_dir=str(args.out_dir / "logs"),
        )


if __name__ == "__main__":
    main()
