#!/usr/bin/env python3
"""Audit-grade forward-only complexity metrics for frozen paper configurations.

This script counts all forward calls used by the training loop:

    train forward per mini-batch + dev forward per mini-batch + test forward per mini-batch

For practicality, FLOPs are profiled once per exact batch signature
(split, stage, train/eval mode, tensor shapes) and multiplied by the number of
times that signature occurs across the completed training run.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import measure_fixed_cases as base

try:
    from math import prod
except ImportError:  # pragma: no cover
    from numpy import prod  # type: ignore


HERE = Path(__file__).resolve().parent
MY_CREATION = HERE.parent
DEFAULT_CASES = ("mosi", "mosei")
DEFAULT_JSON = HERE / "results_audit_forward_only_20260526.json"
DEFAULT_MD = HERE / "results_audit_forward_only_20260526.md"
FULLTRAIN_LOG_GLOB = str(HERE / "nohup_fulltrain_serial_*.log")


@dataclass(frozen=True)
class BatchSignature:
    split: str
    stage: int
    mode: str
    input_ids: tuple[int, ...]
    visual: tuple[int, ...]
    acoustic: tuple[int, ...]
    labels: tuple[int, ...]
    input_mask: tuple[int, ...] | None
    segment_ids: tuple[int, ...] | None

    @property
    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _set_quiet_tqdm(ft) -> Any:
    old_tqdm = getattr(ft, "tqdm", None)
    ft.tqdm = base._quiet_tqdm
    return old_tqdm


def _restore_tqdm(ft, old_tqdm: Any) -> None:
    if old_tqdm is not None:
        ft.tqdm = old_tqdm


def _shape(x: torch.Tensor | None) -> tuple[int, ...] | None:
    if x is None:
        return None
    return tuple(int(v) for v in x.shape)


def _make_signature(
    split: str,
    stage: int,
    mode: str,
    inputs: tuple[torch.Tensor, ...],
) -> BatchSignature:
    input_ids, visual, acoustic, labels = inputs[:4]
    input_mask = inputs[4] if len(inputs) >= 5 else None
    segment_ids = inputs[5] if len(inputs) >= 6 else None
    return BatchSignature(
        split=split,
        stage=int(stage),
        mode=mode,
        input_ids=_shape(input_ids) or (),
        visual=_shape(visual) or (),
        acoustic=_shape(acoustic) or (),
        labels=_shape(labels) or (),
        input_mask=_shape(input_mask),
        segment_ids=_shape(segment_ids),
    )


def _loader_inputs(ft, batch) -> tuple[torch.Tensor, ...]:
    input_ids, visual, acoustic, label_ids, input_mask, segment_ids = base._unpack_loader_batch(
        ft,
        batch,
        ft.DEVICE,
    )
    parts: list[torch.Tensor] = [input_ids, visual, acoustic, label_ids]
    if input_mask is not None:
        parts.append(input_mask)
    if segment_ids is not None:
        parts.append(segment_ids)
    return tuple(parts)


def _collect_one_epoch_signatures(ft, loaders: dict[str, Any]) -> dict[str, Any]:
    """Collect exact per-epoch signature counts and one sample tensor tuple per signature."""
    split_modes = {"train": "train", "dev": "eval", "test": "eval"}
    stages = (1, 2)
    counts: Counter[str] = Counter()
    samples: dict[str, tuple[torch.Tensor, ...]] = {}
    signatures: dict[str, BatchSignature] = {}
    per_split_batches: dict[str, int] = {}
    per_split_signature_counts: dict[str, Counter[str]] = {}

    for split, loader in loaders.items():
        split_counter: Counter[str] = Counter()
        split_batches = 0
        for batch in loader:
            split_batches += 1
            inputs = _loader_inputs(ft, batch)
            for stage in stages:
                sig = _make_signature(split, stage, split_modes[split], inputs)
                counts[sig.key] += 1
                split_counter[sig.key] += 1
                signatures[sig.key] = sig
                samples.setdefault(sig.key, inputs)
        per_split_batches[split] = split_batches
        per_split_signature_counts[split] = split_counter

    return {
        "counts": counts,
        "samples": samples,
        "signatures": signatures,
        "per_split_batches": per_split_batches,
        "per_split_signature_counts": per_split_signature_counts,
    }


def _parse_fulltrain_logs(case: str) -> dict[str, Any] | None:
    pattern = re.compile(
        rf"\[full-training\]\s+{re.escape(case)}\s+epoch\s+(\d+)/(\d+):\s+([0-9.]+)s",
    )
    candidates: list[dict[str, Any]] = []
    for path_s in glob.glob(FULLTRAIN_LOG_GLOB):
        path = Path(path_s)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        epoch_secs = [(int(m.group(1)), int(m.group(2)), float(m.group(3))) for m in pattern.finditer(text)]
        if not epoch_secs:
            continue
        epochs = [e for e, _total, _sec in epoch_secs]
        secs = [sec for _e, _total, sec in epoch_secs]
        candidates.append(
            {
                "source": str(path),
                "mtime": path.stat().st_mtime,
                "epochs_completed": max(epochs),
                "n_epochs_config": epoch_secs[-1][1],
                "epoch_secs": secs,
                "avg_epoch_sec": sum(secs) / len(secs),
            },
        )
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x["epochs_completed"], x["mtime"]), reverse=True)
    best = candidates[0]
    best.pop("mtime", None)
    return best


def _parse_fulltrain_markdown(case: str) -> dict[str, Any] | None:
    paths = sorted(HERE.glob(f"results_{case}_fulltrain_paper_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        return None
    path = paths[0]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    rows = [ln for ln in lines if ln.startswith(f"| {case} |")]
    if not rows:
        return None
    cells = [c.strip() for c in rows[0].strip("|").split("|")]
    # case, params, trainable, avg_epoch_FLOPs_est, peak_GPU_GiB,
    # avg_epoch_sec, train_total_sec, epochs_completed, ...
    try:
        return {
            "source": str(path),
            "peak_memory_training_loop_gib": float(cells[4]),
            "avg_epoch_sec": float(cells[5]),
            "train_total_sec": float(cells[6]),
            "epochs_completed": int(cells[7]),
        }
    except (IndexError, ValueError):
        return {"source": str(path)}


def _completed_epoch_info(case: str, configured_epochs: int) -> dict[str, Any]:
    from_log = _parse_fulltrain_logs(case)
    from_md = _parse_fulltrain_markdown(case)
    if from_log:
        out = {
            "epochs_completed": int(from_log["epochs_completed"]),
            "n_epochs_config": int(from_log["n_epochs_config"]),
            "avg_complete_epoch_sec": float(from_log["avg_epoch_sec"]),
            "epoch_secs": from_log["epoch_secs"],
            "time_source": from_log["source"],
        }
        if from_md and "peak_memory_training_loop_gib" in from_md:
            out["peak_memory_training_loop_gib"] = from_md["peak_memory_training_loop_gib"]
            out["training_memory_source"] = from_md["source"]
        return out
    if from_md and "epochs_completed" in from_md:
        return {
            "epochs_completed": int(from_md["epochs_completed"]),
            "n_epochs_config": int(configured_epochs),
            "avg_complete_epoch_sec": from_md.get("avg_epoch_sec"),
            "epoch_secs": None,
            "time_source": from_md["source"],
            "peak_memory_training_loop_gib": from_md.get("peak_memory_training_loop_gib"),
            "training_memory_source": from_md["source"],
        }
    return {
        "epochs_completed": int(configured_epochs),
        "n_epochs_config": int(configured_epochs),
        "avg_complete_epoch_sec": None,
        "epoch_secs": None,
        "time_source": "configured fallback; no full-training timing log found",
    }


def _epoch_stage_counts(epochs_completed: int, stage1_epochs: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for epoch in range(1, int(epochs_completed) + 1):
        stage = 1 if epoch <= int(stage1_epochs) else 2
        counts[stage] += 1
    return counts


def _jit_shape(value: Any) -> list[int] | None:
    from fvcore.nn.jit_handles import get_shape  # type: ignore

    shape = get_shape(value)
    if shape is None:
        return None
    return [int(v) for v in shape if v is not None]


def _numel_from_shape(shape: list[int] | None) -> int:
    if not shape:
        return 0
    return int(prod(shape))


def _out_numel(_inputs: list[Any], outputs: list[Any], multiplier: float = 1.0) -> float:
    return float(_numel_from_shape(_jit_shape(outputs[0]))) * float(multiplier)


def _in_numel(inputs: list[Any], _outputs: list[Any], idx: int = 0, multiplier: float = 1.0) -> float:
    return float(_numel_from_shape(_jit_shape(inputs[idx]))) * float(multiplier)


def _cosine_similarity_flops(inputs: list[Any], _outputs: list[Any]) -> float:
    shape = _jit_shape(inputs[0])
    if not shape:
        return 0.0
    # mul + two square/norm paths + sum + max/eps + final div, under the same
    # one-FLOP-per-element convention used for other elementwise audit handles.
    return float(_numel_from_shape(shape)) * 6.0


def _argsort_flops(inputs: list[Any], _outputs: list[Any]) -> float:
    shape = _jit_shape(inputs[0])
    if not shape:
        return 0.0
    n = int(shape[-1]) if shape else 0
    return float(_numel_from_shape(shape)) * math.log2(max(n, 2))


def _install_audit_op_handles(analysis: Any) -> Any:
    """Account for common transformer/InfoGate ops missing from fvcore defaults.

    fvcore already handles linear/matmul/bmm. These handles add explicit
    arithmetic counts for elementwise, activation, softmax, reduction, and
    routing-helper ops that otherwise appear as unsupported. Embedding and pure
    indexing/masking ops are assigned zero FLOPs because they are memory/index
    operations under the arithmetic-FLOP convention used here.
    """

    zero_ops = [
        "aten::embedding",
        "aten::broadcast_tensors",
        "aten::empty_like",
        "aten::randn_like",
        "aten::exponential_",
        "aten::scatter_",
        "aten::one_hot",
        "aten::argmax",
        "aten::masked_select",
    ]
    one_out_ops = [
        "aten::ne",
        "aten::mul",
        "aten::sub",
        "aten::sign",
        "aten::lt",
        "aten::abs",
        "aten::where",
        "aten::div",
        "aten::ceil",
        "aten::add",
        "aten::le",
        "aten::sqrt",
        "aten::neg",
        "aten::add_",
        "aten::sigmoid",
        "aten::clamp_min",
        "aten::pow",
        "aten::rsub",
        "prim::PythonOp.XDropout",
    ]
    reduction_ops = [
        "aten::sum",
        "aten::mean",
        "aten::min",
        "aten::l1_loss",
        "aten::kl_div",
    ]
    for op in zero_ops:
        analysis = analysis.set_op_handle(op, lambda _i, _o: 0.0)
    for op in one_out_ops:
        analysis = analysis.set_op_handle(op, lambda i, o: _out_numel(i, o))
    for op in reduction_ops:
        analysis = analysis.set_op_handle(op, lambda i, o: _in_numel(i, o))
    analysis = analysis.set_op_handle("aten::log", lambda i, o: _out_numel(i, o, 4.0))
    analysis = analysis.set_op_handle("aten::exp", lambda i, o: _out_numel(i, o, 4.0))
    analysis = analysis.set_op_handle("aten::gelu", lambda i, o: _out_numel(i, o, 6.0))
    analysis = analysis.set_op_handle("aten::softmax", lambda i, o: _out_numel(i, o, 5.0))
    analysis = analysis.set_op_handle("prim::PythonOp.XSoftmax", lambda i, o: _out_numel(i, o, 5.0))
    analysis = analysis.set_op_handle("aten::repeat", lambda i, o: 0.0)
    analysis = analysis.set_op_handle("aten::argsort", _argsort_flops)
    analysis = analysis.set_op_handle("aten::cosine_similarity", _cosine_similarity_flops)
    return analysis


def _count_forward_flops_audit(
    wrapper: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    train_mode: bool,
) -> tuple[float | None, str, str | None, dict[str, int]]:
    err: str | None = None
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        was_training = wrapper.training
        wrapper.train(train_mode)
        try:
            analysis = FlopCountAnalysis(wrapper, inputs)
            analysis = _install_audit_op_handles(analysis)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            total = float(analysis.total())
            unsupported = {str(k): int(v) for k, v in analysis.unsupported_ops().items()}
        finally:
            wrapper.train(was_training)
        return total, "fvcore+audit_handles", None, unsupported
    except Exception as exc:  # noqa: BLE001
        err = f"fvcore+audit_handles:{exc!r}"
    return None, "none", err, {}


def _profile_signature(
    ft,
    model,
    sig: BatchSignature,
    inputs: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    train_mode = sig.mode == "train"
    wrapper = base._FlopMeasuringWrapper(ft, model, stage=sig.stage)
    flops, backend, error, unsupported = _count_forward_flops_audit(
        wrapper,
        inputs,
        train_mode=train_mode,
    )
    return {
        "flops_forward": flops,
        "backend": backend,
        "error": error,
        "unsupported_ops": unsupported,
    }


def _measure_forward_peak_memory(
    ft,
    model,
    sig: BatchSignature,
    inputs: tuple[torch.Tensor, ...],
) -> float | None:
    if not torch.cuda.is_available():
        return None
    train_mode = sig.mode == "train"
    wrapper = base._FlopMeasuringWrapper(ft, model, stage=sig.stage)
    wrapper.train(train_mode)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.enable_grad() if train_mode else torch.no_grad():
        out = wrapper(*inputs)
        if train_mode:
            # Keep the graph-building cost of a real training forward, but do not
            # include backward FLOPs or backward memory.
            _ = out.detach()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    del out
    torch.cuda.empty_cache()
    return peak


def _profile_all_signatures(
    ft,
    model,
    collected: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for key, sig in collected["signatures"].items():
        profile = _profile_signature(ft, model, sig, collected["samples"][key])
        profile["peak_memory_forward_only_gib"] = _measure_forward_peak_memory(
            ft,
            model,
            sig,
            collected["samples"][key],
        )
        profiles[key] = profile
    return profiles


def _combine_backend(profiles: dict[str, dict[str, Any]]) -> str:
    backends = {str(v.get("backend")) for v in profiles.values() if v.get("backend") != "none"}
    if not backends:
        return "none"
    if len(backends) == 1:
        return next(iter(backends))
    return "mixed:" + ",".join(sorted(backends))


def _environment() -> dict[str, Any]:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _audit_case(case: str) -> dict[str, Any]:
    if str(MY_CREATION) not in sys.path:
        sys.path.insert(0, str(MY_CREATION))

    spec = base._CASE_TO_SPEC[case]
    build_train_argv = base._load_build_train_argv(spec["hparams"])

    with tempfile.TemporaryDirectory(dir=str(base._tmp_root())) as tmp:
        sys.argv = [str(spec["argv0"]), *build_train_argv(checkpoint_dir=tmp)]
        ft = base._load_train_module(spec["train_py"])
        old_tqdm = _set_quiet_tqdm(ft)
        try:
            ft.set_seed(ft.args.seed)
            train_dl, dev_dl, test_dl, n_opt = ft.setup_data()
            model, optimizer, scheduler = ft.build_model(n_opt)
            del optimizer, scheduler

            params_total = int(sum(p.numel() for p in model.parameters()))
            params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            epoch_info = _completed_epoch_info(case, int(ft.args.n_epochs))
            stage_counts = _epoch_stage_counts(
                int(epoch_info["epochs_completed"]),
                int(ft.args.stage1_epochs),
            )
            collected = _collect_one_epoch_signatures(
                ft,
                {"train": train_dl, "dev": dev_dl, "test": test_dl},
            )
            profiles = _profile_all_signatures(ft, model, collected)
            unsupported_ops: Counter[str] = Counter()
            for profile in profiles.values():
                unsupported_ops.update(profile.get("unsupported_ops", {}))

            signature_details: list[dict[str, Any]] = []
            total_forward_flops = 0.0
            total_forward_calls = 0
            split_totals: dict[str, dict[str, Any]] = {}
            max_forward_memory = 0.0

            for split, one_epoch_counter in collected["per_split_signature_counts"].items():
                split_flops = 0.0
                split_calls = 0
                for key, per_epoch_count in one_epoch_counter.items():
                    sig = collected["signatures"][key]
                    epochs_for_stage = stage_counts[int(sig.stage)]
                    total_count = int(per_epoch_count) * int(epochs_for_stage)
                    if total_count == 0:
                        continue
                    profile = profiles[key]
                    flops = profile.get("flops_forward")
                    if flops is None:
                        raise RuntimeError(f"FLOP profiling failed for {case} {sig}: {profile.get('error')}")
                    contrib = float(flops) * float(total_count)
                    peak_mem = profile.get("peak_memory_forward_only_gib")
                    if peak_mem is not None:
                        max_forward_memory = max(max_forward_memory, float(peak_mem))
                    split_flops += contrib
                    split_calls += total_count
                    total_forward_flops += contrib
                    total_forward_calls += total_count
                    signature_details.append(
                        {
                            **asdict(sig),
                            "count_per_epoch": int(per_epoch_count),
                            "epochs_for_stage": int(epochs_for_stage),
                            "total_forward_calls": total_count,
                            "flops_per_forward": float(flops),
                            "flops_total": contrib,
                            "peak_memory_forward_only_gib": peak_mem,
                            "backend": profile.get("backend"),
                            "unsupported_ops": profile.get("unsupported_ops", {}),
                        },
                    )
                split_totals[split] = {
                    "forward_calls_total": split_calls,
                    "forward_flops_total": split_flops,
                    "avg_epoch_forward_flops": split_flops / float(epoch_info["epochs_completed"]),
                }

            expected_calls = int(epoch_info["epochs_completed"]) * sum(collected["per_split_batches"].values())
            avg_epoch_forward_flops = total_forward_flops / float(epoch_info["epochs_completed"])
            result = {
                "case": case,
                "dataset": ft.args.dataset,
                "params_total": params_total,
                "params_trainable": params_trainable,
                "train_batch_size": int(ft.args.train_batch_size),
                "gradient_accumulation_step": int(ft.args.gradient_accumulation_step),
                "max_seq_length": int(ft.args.max_seq_length),
                "stage1_epochs": int(ft.args.stage1_epochs),
                "n_epochs_config": int(ft.args.n_epochs),
                "epochs_completed": int(epoch_info["epochs_completed"]),
                "stage_epoch_counts": {str(k): int(v) for k, v in sorted(stage_counts.items())},
                "batches_per_epoch": collected["per_split_batches"],
                "forward_calls_total": total_forward_calls,
                "forward_calls_expected": expected_calls,
                "forward_call_count_ok": total_forward_calls == expected_calls,
                "forward_flops_total": total_forward_flops,
                "avg_epoch_forward_flops": avg_epoch_forward_flops,
                "split_totals": split_totals,
                "peak_memory_forward_only_gib": max_forward_memory if torch.cuda.is_available() else None,
                "peak_memory_training_loop_gib": epoch_info.get("peak_memory_training_loop_gib"),
                "avg_complete_epoch_sec": epoch_info.get("avg_complete_epoch_sec"),
                "epoch_secs": epoch_info.get("epoch_secs"),
                "time_source": epoch_info.get("time_source"),
                "training_memory_source": epoch_info.get("training_memory_source"),
                "flops_backend": _combine_backend(profiles),
                "flops_protocol": "forward_only_sum_all_train_dev_test_forwards_by_exact_batch_signature",
                "unsupported_ops_after_audit_handles": dict(sorted(unsupported_ops.items())),
                "signature_details": sorted(
                    signature_details,
                    key=lambda x: (x["split"], int(x["stage"]), str(x["mode"]), str(x["input_ids"])),
                ),
                "environment": _environment(),
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del model, train_dl, dev_dl, test_dl
            return result
        finally:
            _restore_tqdm(ft, old_tqdm)


def _worker_python() -> str:
    return base._worker_python()


def _run_worker(case: str) -> dict[str, Any]:
    cmd = [
        _worker_python(),
        "-u",
        str(Path(__file__).resolve()),
        "--measure-worker",
        case,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(MY_CREATION),
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"audit worker {case} failed rc={proc.returncode}\nstdout:\n{proc.stdout}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"audit worker {case}: empty stdout")
    return json.loads(lines[-1])


def _fmt_int(x: int | float | None) -> str:
    if x is None:
        return "—"
    return f"{int(round(float(x))):,}"


def _fmt_flops(x: float | None) -> str:
    if x is None:
        return "—"
    xf = float(x)
    if xf >= 1e12:
        return f"{xf / 1e12:.3f}T"
    if xf >= 1e9:
        return f"{xf / 1e9:.3f}G"
    if xf >= 1e6:
        return f"{xf / 1e6:.3f}M"
    return f"{xf:.3e}"


def _fmt_float(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "—"
    return f"{float(x):.{digits}f}"


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Audit-grade forward-only model complexity",
        "",
        "FLOPs are forward-only. For each completed training run, the audit sums every train/dev/test forward call and divides by completed epochs. Batch signatures are cached only when tensor shapes and stage/mode match exactly.",
        "",
        "| case | params | trainable | avg_epoch_forward_FLOPs | total_forward_FLOPs | peak_forward_mem_GiB | peak_training_mem_GiB | avg_complete_epoch_sec | epochs | forward calls | call check | B×accum | seq |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {params} | {trainable} | {avg_flops} | {total_flops} | "
            "{fmem} | {tmem} | {epoch_sec} | {epochs} | {calls} | {ok} | {batch}×{accum} | {seq} |".format(
                case=row["case"],
                params=_fmt_int(row["params_total"]),
                trainable=_fmt_int(row["params_trainable"]),
                avg_flops=_fmt_flops(row["avg_epoch_forward_flops"]),
                total_flops=_fmt_flops(row["forward_flops_total"]),
                fmem=_fmt_float(row.get("peak_memory_forward_only_gib")),
                tmem=_fmt_float(row.get("peak_memory_training_loop_gib")),
                epoch_sec=_fmt_float(row.get("avg_complete_epoch_sec"), 2),
                epochs=_fmt_int(row.get("epochs_completed")),
                calls=_fmt_int(row.get("forward_calls_total")),
                ok="OK" if row.get("forward_call_count_ok") else "CHECK",
                batch=row["train_batch_size"],
                accum=row["gradient_accumulation_step"],
                seq=row["max_seq_length"],
            ),
        )

    lines.extend(["", "## Per-case audit details", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['case']}",
                "",
                f"- FLOP backend: `{row['flops_backend']}`",
                f"- Protocol: `{row['flops_protocol']}`",
                f"- Unsupported ops after audit handles: `{row.get('unsupported_ops_after_audit_handles', {})}`",
                f"- Stage epoch counts: `{row['stage_epoch_counts']}`",
                f"- Batches per epoch: `{row['batches_per_epoch']}`",
                f"- Time source: `{row.get('time_source')}`",
                f"- Training memory source: `{row.get('training_memory_source', 'not available')}`",
                "",
                "| split | total forward calls | total forward FLOPs | avg epoch forward FLOPs |",
                "|---|---:|---:|---:|",
            ],
        )
        for split, stats in row["split_totals"].items():
            lines.append(
                f"| {split} | {_fmt_int(stats['forward_calls_total'])} | "
                f"{_fmt_flops(stats['forward_flops_total'])} | "
                f"{_fmt_flops(stats['avg_epoch_forward_flops'])} |",
            )
        lines.extend(["", "| split | stage | mode | input shape | calls | FLOPs/forward | peak forward GiB |", "|---|---:|---|---|---:|---:|---:|"])
        for sig in row["signature_details"]:
            lines.append(
                f"| {sig['split']} | {sig['stage']} | {sig['mode']} | "
                f"`{sig['input_ids']}` | {_fmt_int(sig['total_forward_calls'])} | "
                f"{_fmt_flops(sig['flops_per_forward'])} | "
                f"{_fmt_float(sig.get('peak_memory_forward_only_gib'))} |",
            )
        lines.append("")
    return "\n".join(lines)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "params_total",
        "params_trainable",
        "avg_epoch_forward_flops",
        "forward_flops_total",
        "peak_memory_forward_only_gib",
        "peak_memory_training_loop_gib",
        "avg_complete_epoch_sec",
        "epochs_completed",
        "forward_calls_total",
        "forward_calls_expected",
        "forward_call_count_ok",
        "train_batch_size",
        "gradient_accumulation_step",
        "max_seq_length",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure-worker", help=argparse.SUPPRESS)
    ap.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated subset of mosi,mosei.")
    ap.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--output-csv", type=Path, default=HERE / "results_audit_forward_only_20260526.csv")
    args = ap.parse_args()

    if args.measure_worker:
        case = args.measure_worker.strip().lower()
        if case not in DEFAULT_CASES:
            print(f"unknown audit case {case!r}", file=sys.stderr)
            return 2
        result = _audit_case(case)
        print(json.dumps(result), flush=True)
        return 0

    cases = [c.strip().lower() for c in args.cases.split(",") if c.strip()]
    for case in cases:
        if case not in DEFAULT_CASES:
            print(f"unknown case in --cases: {case!r}", file=sys.stderr)
            return 2

    started = time.time()
    rows = [_run_worker(case) for case in cases]
    payload = {
        "generated_at_unix": started,
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(started)),
        "cases": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(_markdown(rows), encoding="utf-8")
    _write_csv(rows, args.output_csv)
    print(_markdown(rows))
    print(f"wrote {args.output_json}", file=sys.stderr)
    print(f"wrote {args.output_md}", file=sys.stderr)
    print(f"wrote {args.output_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
