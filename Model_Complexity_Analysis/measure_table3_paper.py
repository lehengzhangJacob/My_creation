#!/usr/bin/env python3
"""Recompute paper Table (model complexity) for MOSI / MOSEI / SIMS-v2.

Uses the Nov-2026 paper configs wired in ``measure_fixed_cases._CASE_TO_SPEC``.
Reports the same rows as ``tab:model_complexity`` plus a SIMS-v2 column.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import measure_fixed_cases as base

HERE = Path(__file__).resolve().parent


def _fmt_params(n: int) -> str:
    return f"{n / 1e6:.1f}M"


def _fmt_flops(x: float | None) -> str:
    if x is None:
        return "n/a"
    ax = abs(x)
    if ax >= 1e12:
        return f"{x / 1e12:.2f}T"
    if ax >= 1e9:
        return f"{x / 1e9:.2f}G"
    if ax >= 1e6:
        return f"{x / 1e6:.2f}M"
    return f"{x:.3g}"


def measure_case(case: str) -> dict:
    # Run in-process (not subprocess) so CUDA_VISIBLE_DEVICES from the launcher sticks.
    row = base.measure_worker(
        case,
        memory_full_accum=True,
        skip_epoch_time=True,
        full_training_time=False,
    )
    # Re-open a fresh worker-style session for eval FLOPs + epoch timing.
    # measure_worker already tore down; call a richer inlined path.
    return _measure_rich(case, peak_from_worker=row)


def _measure_rich(case: str, peak_from_worker: dict) -> dict:
    import os
    import tempfile

    os.environ.setdefault("TQDM_DISABLE", "1")
    if str(base._MY_CREATION) not in sys.path:
        sys.path.insert(0, str(base._MY_CREATION))

    spec = base._CASE_TO_SPEC[case]
    build = base._load_build_train_argv(spec["hparams"])
    with tempfile.TemporaryDirectory(dir=str(base._tmp_root())) as tmp:
        argv = build(checkpoint_dir=tmp)
        sys.argv = [str(spec["argv0"]), *argv]
        ft = base._load_train_module(spec["train_py"])
        args = ft.args
        train_dl, dev_dl, test_dl, n_opt = ft.setup_data()
        model, optimizer, scheduler = ft.build_model(n_opt)

        params = sum(p.numel() for p in model.parameters())

        # Match prior paper audit: stage-1 train fwd (train mode) + stage-2 eval fwd.
        # If eval B=128 OOMs, fall back to a smaller probe batch and scale FLOPs ~linearly in B.
        def _profile_eval(loader):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            prof = base._profile_forward_flops(ft, model, loader, stage=2, train_mode=False)
            if prof.get("flops") is not None:
                return prof
            # OOM / fail: rebuild a micro-loader with train batch size and scale.
            probe_bs = max(1, int(args.train_batch_size))
            target_bs = int(getattr(args, "dev_batch_size", probe_bs) if loader is dev_dl else getattr(args, "test_batch_size", probe_bs))
            print(f"[warn] eval FLOPs failed ({prof.get('error')}); probing bs={probe_bs} scale→{target_bs}", flush=True)
            from torch.utils.data import DataLoader
            ds = loader.dataset
            probe_dl = DataLoader(ds, batch_size=probe_bs, shuffle=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            probe = base._profile_forward_flops(ft, model, probe_dl, stage=2, train_mode=False)
            fl = probe.get("flops")
            if fl is None:
                return probe
            scale = float(target_bs) / float(probe_bs)
            return {
                "flops": fl * scale,
                "backend": f"{probe.get('backend')}+bs_scale_{probe_bs}to{target_bs}",
                "error": None,
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        train_prof = base._profile_forward_flops(ft, model, train_dl, stage=1, train_mode=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dev_prof = _profile_eval(dev_dl)
        test_prof = _profile_eval(test_dl)

        n_train = len(train_dl)
        n_dev = len(dev_dl)
        n_test = len(test_dl)

        flops_train = train_prof.get("flops")
        flops_dev = dev_prof.get("flops")
        flops_test = test_prof.get("flops")

        # Peak over one full loop epoch (train+dev+test), closer to paper full-train peak.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ft.train_epoch(model, train_dl, optimizer, scheduler, stage=1, epoch_idx=1, ema=None)
        ft.test_epoch(model, dev_dl, stage=1, desc="Dev")
        ft.test_epoch(model, test_dl, stage=1, desc="Test")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_sec = time.perf_counter() - t0
        peak = (
            torch.cuda.max_memory_allocated() / (1024.0 ** 3)
            if torch.cuda.is_available()
            else peak_from_worker.get("peak_mem_gib")
        )

        out = {
            "case": case,
            "params": params,
            "flops_train_fwd": flops_train,
            "flops_train_epoch": (flops_train * n_train) if flops_train else None,
            "flops_dev_epoch": (flops_dev * n_dev) if flops_dev else None,
            "flops_test_epoch": (flops_test * n_test) if flops_test else None,
            "peak_mem_gib": peak,
            "epoch_sec": epoch_sec,
            "n_train": n_train,
            "n_dev": n_dev,
            "n_test": n_test,
            "batch": int(args.train_batch_size),
            "accum": int(args.gradient_accumulation_step),
            "seq": int(args.max_seq_length),
            "flop_backend": train_prof.get("backend"),
            "flop_error": train_prof.get("error"),
        }
        del model, optimizer, scheduler, train_dl, dev_dl, test_dl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out


def latex_table(rows: list[dict]) -> str:
    order = ["mosi", "mosei", "simsv2"]
    by = {r["case"]: r for r in rows}
    cols = [by[c] for c in order if c in by]
    headers = {"mosi": "MOSI", "mosei": "MOSEI", "simsv2": "SIMS-v2"}
    head = " & ".join(headers[c["case"]] for c in cols)

    def cell(key, fmt):
        return " & ".join(fmt(c[key]) for c in cols)

    lines = [
        r"\begin{tabular}{l|" + "c" * len(cols) + "}",
        r"\toprule",
        f"& {head} \\\\",
        r"\midrule",
        f"Parameters & {cell('params', _fmt_params)} \\\\",
        f"FLOPs / train fwd & {cell('flops_train_fwd', _fmt_flops)} \\\\",
        f"Train fwd FLOPs / epoch & {cell('flops_train_epoch', _fmt_flops)} \\\\",
        f"Dev eval FLOPs / epoch & {cell('flops_dev_epoch', _fmt_flops)} \\\\",
        f"Test eval FLOPs / epoch & {cell('flops_test_epoch', _fmt_flops)} \\\\",
        "Peak train memory (GiB) & "
        + " & ".join(f"{c['peak_mem_gib']:.3f}" if c.get("peak_mem_gib") is not None else "n/a" for c in cols)
        + r" \\",
        "Avg. epoch time (full loop, s) & "
        + " & ".join(f"{c['epoch_sec']:.2f}" if c.get("epoch_sec") is not None else "n/a" for c in cols)
        + r" \\",
        "Batches / epoch & "
        + " & ".join(
            f"{c['n_train']:,}".replace(",", "{,}") if c["n_train"] >= 1000 else str(c["n_train"])
            for c in cols
        )
        + r" \\",
        "Batch $\\times$ accum. & "
        + " & ".join(f"{c['batch']}$\\times${c['accum']}" for c in cols)
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    # Fix latex thousands: use 4{,}067 style
    fixed = []
    for ln in lines:
        if "Batches / epoch" in ln:
            parts = []
            for c in cols:
                n = c["n_train"]
                parts.append(f"{n:,}".replace(",", "{,}"))
            ln = "Batches / epoch & " + " & ".join(parts) + r" \\"
        fixed.append(ln)
    return "\n".join(fixed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="mosi,mosei,simsv2")
    ap.add_argument("--output-json", type=Path, default=HERE / "results_table3_2026_11.json")
    ap.add_argument("--output-tex", type=Path, default=HERE / "results_table3_2026_11.tex")
    args = ap.parse_args()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    rows = []
    for case in cases:
        print(f"=== measuring {case} ===", flush=True)
        row = _measure_rich(case, peak_from_worker={})
        print(json.dumps({k: row[k] for k in row if k != "flop_error"}, indent=2), flush=True)
        rows.append(row)
    args.output_json.write_text(json.dumps(rows, indent=2) + "\n")
    tex = latex_table(rows)
    args.output_tex.write_text(tex + "\n")
    print(tex)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
