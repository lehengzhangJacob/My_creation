#!/usr/bin/env python3
"""Model complexity metrics for paper frozen configurations (subprocess per case).

Defaults to conda env **ITHP5090**: per-case ``--measure-worker`` subprocesses use
that interpreter (see ``_worker_python()``). Set env **MODEL_COMPLEXITY_PYTHON** to
override. Launch the driver with **ITHP5090** as well (``run_measure.sh`` or the full
``python`` path below) so imports match workers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_MY_CREATION = _HERE.parent
_FIXED_EXP = _MY_CREATION / "fixed_experiment"
_ABLATION = _MY_CREATION / "ablation_study"

# Same default interpreter as ``run_optuna_4090d_restart.sh`` on this project layout.
_DEFAULT_ITHP5090_PY = Path("/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python")


def _worker_python() -> str:
    """Python for ``--measure-worker`` subprocesses; defaults to ITHP5090."""
    override = os.environ.get("MODEL_COMPLEXITY_PYTHON", "").strip()
    if override and Path(override).is_file():
        return override
    if _DEFAULT_ITHP5090_PY.is_file():
        return str(_DEFAULT_ITHP5090_PY)
    return sys.executable


def _tmp_root() -> Path:
    """Keep temporary checkpoints off the small system /tmp partition."""
    root = (
        os.environ.get("MODEL_COMPLEXITY_TMPDIR")
        or os.environ.get("TMPDIR")
        or str(_HERE / "tmp")
    )
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


_CASE_TO_SPEC = {
    # Current paper rows: MOSI more/mosi trial 39 and MOSEI phase1 trial 70.
    "mosi": {
        "hparams": _ABLATION / "mosi_more_trial39_hparams.py",
        "train_py": _ABLATION / "train.py",
        "argv0": "ablation_study/train.py",
    },
    "mosei": {
        "hparams": _ABLATION / "mosei_phase1_trial70_hparams.py",
        "train_py": _ABLATION / "train.py",
        "argv0": "ablation_study/train.py",
    },
    "simsv2": {
        "hparams": _FIXED_EXP / "simsv2_phase4_trial52_hparams.py",
        "train_py": _FIXED_EXP / "train.py",
        "argv0": "fixed_experiment/train.py",
    },
}


def _load_build_train_argv(hp_py: Path):
    spec = importlib.util.spec_from_file_location(
        f"_hparams_{hp_py.stem}", hp_py,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.build_train_argv


def _load_train_module(train_py: Path):
    """Parse CLI from sys.argv and load the selected train.py (must run after sys.argv is set)."""
    spec = importlib.util.spec_from_file_location(
        f"{train_py.parent.name}_train_meas", train_py,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _simsv2_mmsa_enabled(ft) -> bool:
    fn = getattr(ft, "simsv2_mmsa_enabled", None)
    return bool(fn()) if callable(fn) else False


def _unpack_loader_batch(ft, batch, device: torch.device):
    batch = tuple(t.to(device) for t in batch)
    unpack = getattr(ft, "unpack_simsv2_batch", None)
    if callable(unpack):
        input_ids, visual, acoustic, label_ids, input_mask, segment_ids = unpack(
            batch,
            _simsv2_mmsa_enabled(ft),
        )
    else:
        input_ids, visual, acoustic, label_ids = batch[:4]
        input_mask, segment_ids = None, None
    visual = visual.squeeze(1)
    acoustic = acoustic.squeeze(1)
    return input_ids, visual, acoustic, label_ids, input_mask, segment_ids


def _forward_model(ft, model, input_ids, visual, acoustic, label_ids, stage: int, input_mask=None, segment_ids=None):
    fwd = getattr(ft, "_forward_model", None)
    if callable(fwd):
        return fwd(model, input_ids, visual, acoustic, label_ids, stage, input_mask, segment_ids)

    kwargs = {"labels": label_ids, "stage": stage}
    if input_mask is not None:
        kwargs["attention_mask"] = input_mask
    if segment_ids is not None:
        kwargs["token_type_ids"] = segment_ids
    return model(input_ids, visual, acoustic, **kwargs)


class _FlopMeasuringWrapper(nn.Module):
    """Expose one forward for fvcore/thop at a specific training stage."""

    def __init__(self, ft, core: nn.Module, stage: int) -> None:
        super().__init__()
        self.ft = ft
        self.core = core
        self.stage = int(stage)

    def forward(
        self,
        input_ids: torch.Tensor,
        visual: torch.Tensor,
        acoustic: torch.Tensor,
        labels: torch.Tensor,
        *extra: torch.Tensor,
    ) -> torch.Tensor:
        input_mask = extra[0] if len(extra) >= 1 else None
        segment_ids = extra[1] if len(extra) >= 2 else None
        logits, ib_loss, _, _ = _forward_model(
            self.ft,
            self.core,
            input_ids,
            visual,
            acoustic,
            labels,
            self.stage,
            input_mask,
            segment_ids,
        )
        return logits.sum() + ib_loss


def _count_flops(
    wrapper: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    train_mode: bool = True,
) -> tuple[float | None, str, str | None]:
    err: str | None = None
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        was_training = wrapper.training
        wrapper.train(train_mode)
        try:
            total = FlopCountAnalysis(wrapper, inputs).total()
        finally:
            wrapper.train(was_training)
        return float(total), "fvcore", None
    except Exception as e:  # noqa: BLE001
        err = f"fvcore:{e!r}"
    try:
        from thop import profile  # type: ignore

        was_training = wrapper.training
        wrapper.train(train_mode)
        try:
            macs, _params = profile(wrapper, inputs=inputs, verbose=False)
            flops = 2.0 * float(macs)
            return flops, "thop", None
        finally:
            wrapper.train(was_training)
    except Exception as e2:  # noqa: BLE001
        err = (err or "") + f"|thop:{e2!r}"
    return None, "none", err


def _microbatch_from_loader(ft, train_dl, device: torch.device):
    batch = next(iter(train_dl))
    return _unpack_loader_batch(ft, batch, device)


def _one_step_loss(
    ft,
    model,
    args,
    input_ids,
    visual,
    acoustic,
    label_ids,
    input_mask,
    segment_ids,
    stage: int,
):
    logits, ib_loss, loss_dict, _ = _forward_model(
        ft,
        model,
        input_ids,
        visual,
        acoustic,
        label_ids,
        stage,
        input_mask,
        segment_ids,
    )
    pred_flat = logits.view(-1)
    label_flat = label_ids.view(-1)
    loss_dict["pred_mean"] = pred_flat.mean().item()
    loss_dict["pred_std"] = pred_flat.std(unbiased=False).item()
    l_task = F.l1_loss(pred_flat, label_flat) + args.mse_weight * F.mse_loss(
        pred_flat, label_flat,
    )
    loss = l_task + ib_loss
    if args.gradient_accumulation_step > 1:
        loss = loss / args.gradient_accumulation_step
    return loss


def _peak_memory_microbatch(ft, model, optimizer, train_dl) -> float | None:
    if not torch.cuda.is_available():
        return None
    device = ft.DEVICE
    model.train()
    optimizer.zero_grad()
    input_ids, visual, acoustic, label_ids, input_mask, segment_ids = _unpack_loader_batch(
        ft, next(iter(train_dl)), device,
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    loss = _one_step_loss(
        ft, model, ft.args, input_ids, visual, acoustic, label_ids, input_mask, segment_ids, stage=1,
    )
    loss.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024.0 ** 3)


def _peak_memory_accum_window(ft, model, optimizer, train_dl) -> float | None:
    if not torch.cuda.is_available():
        return None
    device = ft.DEVICE
    args = ft.args
    model.train()
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    it = iter(train_dl)
    ga = max(1, int(args.gradient_accumulation_step))
    for step in range(ga):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl)
            batch = next(it)
        input_ids, visual, acoustic, label_ids, input_mask, segment_ids = _unpack_loader_batch(
            ft, batch, device,
        )
        loss = _one_step_loss(
            ft, model, ft.args, input_ids, visual, acoustic, label_ids, input_mask, segment_ids, stage=1,
        )
        loss.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024.0 ** 3)


def _flop_inputs_from_loader(ft, loader) -> tuple[torch.Tensor, ...]:
    device = ft.DEVICE
    input_ids, visual, acoustic, label_ids, input_mask, segment_ids = _unpack_loader_batch(
        ft,
        next(iter(loader)),
        device,
    )
    parts: list[torch.Tensor] = [input_ids, visual, acoustic, label_ids]
    if input_mask is not None:
        parts.append(input_mask)
    if segment_ids is not None:
        parts.append(segment_ids)
    return tuple(parts)


def _profile_forward_flops(ft, model, loader, *, stage: int, train_mode: bool) -> dict[str, Any]:
    wrapper = _FlopMeasuringWrapper(ft, model, stage=stage)
    inputs = _flop_inputs_from_loader(ft, loader)
    flops, backend, err = _count_flops(wrapper, inputs, train_mode=train_mode)
    return {"flops": flops, "backend": backend, "error": err}


def _combine_flop_backend(profiles: dict[str, dict[str, Any]]) -> str:
    backends = {str(p.get("backend")) for p in profiles.values() if p.get("backend") != "none"}
    if not backends:
        return "none"
    if len(backends) == 1:
        return next(iter(backends))
    return "mixed:" + ",".join(sorted(backends))


def _combine_flop_errors(profiles: dict[str, dict[str, Any]]) -> str | None:
    errors = [
        f"{name}:{profile.get('error')}"
        for name, profile in profiles.items()
        if profile.get("error")
    ]
    return "|".join(errors) if errors else None


def _profile_default_flops(ft, model, train_dl) -> dict[str, Any]:
    return _profile_forward_flops(ft, model, train_dl, stage=1, train_mode=True)


def _profile_full_training_flops(ft, model, train_dl, dev_dl, test_dl) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    for stage in (1, 2):
        profiles[f"train_stage{stage}_forward"] = _profile_forward_flops(
            ft, model, train_dl, stage=stage, train_mode=True,
        )
        profiles[f"dev_stage{stage}_forward"] = _profile_forward_flops(
            ft, model, dev_dl, stage=stage, train_mode=False,
        )
        profiles[f"test_stage{stage}_forward"] = _profile_forward_flops(
            ft, model, test_dl, stage=stage, train_mode=False,
        )
    return {
        "profiles": profiles,
        "backend": _combine_flop_backend(profiles),
        "error": _combine_flop_errors(profiles),
    }


def _estimate_full_training_flops(
    flop_profile: dict[str, Any] | None,
    epoch_stages: list[int],
    *,
    train_batches: int,
    dev_batches: int,
    test_batches: int,
) -> dict[str, Any]:
    if not flop_profile or not epoch_stages:
        return {
            "flops_avg_epoch": None,
            "flops_total_training": None,
            "flops_error": None if flop_profile else "missing full-training FLOP profile",
        }

    profiles = flop_profile.get("profiles", {})
    epoch_totals: list[float] = []
    for stage in epoch_stages:
        keys = (
            f"train_stage{stage}_forward",
            f"dev_stage{stage}_forward",
            f"test_stage{stage}_forward",
        )
        try:
            train_fwd = profiles[keys[0]]["flops"]
            dev_fwd = profiles[keys[1]]["flops"]
            test_fwd = profiles[keys[2]]["flops"]
        except KeyError as exc:
            return {
                "flops_avg_epoch": None,
                "flops_total_training": None,
                "flops_error": f"missing profile {exc!r}",
            }
        if train_fwd is None or dev_fwd is None or test_fwd is None:
            return {
                "flops_avg_epoch": None,
                "flops_total_training": None,
                "flops_error": flop_profile.get("error") or "one or more FLOP profiles failed",
            }
        epoch_totals.append(
            3.0 * float(train_fwd) * float(train_batches)
            + float(dev_fwd) * float(dev_batches)
            + float(test_fwd) * float(test_batches)
        )

    total = float(sum(epoch_totals))
    return {
        "flops_avg_epoch": total / float(len(epoch_totals)),
        "flops_total_training": total,
        "flops_error": flop_profile.get("error"),
    }


class _QuietTqdm:
    def __init__(self, iterable, *args, **kwargs) -> None:
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs) -> None:
        return None


def _quiet_tqdm(iterable, *args, **kwargs):
    return _QuietTqdm(iterable)


def _float_metrics(metrics: dict[str, Any] | None) -> dict[str, float] | None:
    if metrics is None:
        return None
    return {str(k): float(v) for k, v in metrics.items()}


def _time_full_training_run(
    ft,
    *,
    flop_profile: dict[str, Any] | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Run the trainer's full epoch loop once and measure complete per-epoch cost."""
    args = ft.args
    ft.set_seed(args.seed)
    train_dl, dev_dl, test_dl, n_opt = ft.setup_data()
    model, optimizer, scheduler = ft.build_model(n_opt)
    ema = ft.EMA(model, decay=args.ema_decay)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"infogate_{args.dataset}_best.pt")
    select_start = args.stage1_epochs
    select_higher_is_better = ft.selection_higher_is_better(args.selection_metric)
    best_selection_score = float("-inf") if select_higher_is_better else float("inf")
    best_selection_tiebreak = None
    best_results = None
    last_test_results = None
    last_completed_epoch = 0
    early_patience = max(0, int(getattr(args, "early_stop_patience", 0) or 0))
    early_min_delta = float(getattr(args, "early_stop_min_delta", 0.0) or 0.0)
    no_improve_epochs = 0
    early_stopped = False
    epoch_secs: list[float] = []
    epoch_stages: list[int] = []
    n_epochs_to_run = int(args.n_epochs)
    if max_epochs is not None:
        n_epochs_to_run = max(1, min(n_epochs_to_run, int(max_epochs)))

    old_tqdm = getattr(ft, "tqdm", None)
    ft.tqdm = _quiet_tqdm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    total_t0 = time.perf_counter()
    try:
        for epoch in range(n_epochs_to_run):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_t0 = time.perf_counter()
            stage = 1 if epoch < args.stage1_epochs else 2
            epoch_stages.append(stage)

            tau_s = getattr(args, "gumbel_tau_start", 1.0)
            tau_e = getattr(args, "gumbel_tau_end", 0.5)
            tau = tau_s + (tau_e - tau_s) * epoch / max(args.n_epochs - 1, 1)
            m = model.module if hasattr(model, "module") else model
            ig = m.dberta.infogate if hasattr(m, "dberta") else m.bert.infogate
            ig.mselector.gumbel_tau = tau

            eval_with_ema = epoch >= args.ema_start_epoch
            if epoch == args.ema_start_epoch:
                ema.reset(model)
            ft.train_epoch(
                model,
                train_dl,
                optimizer,
                scheduler,
                stage,
                epoch + 1,
                ema=ema if eval_with_ema else None,
            )

            if eval_with_ema:
                ema.apply(model)

            dev_preds, dev_labels = ft.test_epoch(model, dev_dl, stage=stage, desc="Dev")
            dev_m = ft.score(dev_preds, dev_labels)
            dev_kw = ft.selection_kwargs(dev_m)
            selection_score = ft.compute_selection_score(args.selection_metric, **dev_kw)
            selection_tiebreak = ft.build_selection_tiebreak(**dev_kw)

            preds, labels = ft.test_epoch(model, test_dl, stage=stage)
            test_m = ft.score(preds, labels)
            last_test_results = test_m

            if epoch >= select_start:
                prev_best_selection_score = best_selection_score
                prev_best_selection_tiebreak = best_selection_tiebreak
                if best_selection_tiebreak is None:
                    should_save = True
                else:
                    better_score = (
                        selection_score > best_selection_score
                        if select_higher_is_better
                        else selection_score < best_selection_score
                    )
                    same_score = abs(selection_score - best_selection_score) <= 1e-12
                    should_save = better_score or (
                        same_score and selection_tiebreak > best_selection_tiebreak
                    )

                if should_save:
                    best_selection_score = selection_score
                    best_selection_tiebreak = selection_tiebreak
                    best_results = test_m
                    save_dict = {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "dev_mae": dev_m["mae"],
                        "dev_corr": dev_m["corr"],
                        "selection_metric": args.selection_metric,
                        "selection_score": selection_score,
                        "ablation": {
                            "arch": args.ablation,
                            "use_l_lib": args.use_l_lib,
                            "use_l_rib": args.use_l_rib,
                        },
                        "test_results": best_results,
                        "args": args,
                    }
                    if eval_with_ema:
                        save_dict["ema_state_dict"] = ema.state_dict()
                    torch.save(save_dict, ckpt_path)

                if early_patience > 0:
                    improved_for_early_stop = should_save
                    if (
                        early_min_delta > 0.0
                        and should_save
                        and prev_best_selection_tiebreak is not None
                    ):
                        same_score_prev = (
                            abs(selection_score - prev_best_selection_score) <= 1e-12
                        )
                        if same_score_prev:
                            improved_for_early_stop = True
                        elif math.isfinite(prev_best_selection_score):
                            if select_higher_is_better:
                                improved_for_early_stop = (
                                    selection_score - prev_best_selection_score
                                ) > early_min_delta
                            else:
                                improved_for_early_stop = (
                                    prev_best_selection_score - selection_score
                                ) > early_min_delta
                        else:
                            improved_for_early_stop = True
                    if improved_for_early_stop:
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1

            if eval_with_ema:
                ema.restore(model)

            last_completed_epoch = epoch + 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_sec = time.perf_counter() - epoch_t0
            epoch_secs.append(epoch_sec)
            print(
                f"[full-training] {args.dataset} epoch {last_completed_epoch}/"
                f"{n_epochs_to_run}: {epoch_sec:.2f}s",
                file=sys.stderr,
                flush=True,
            )

            if (
                early_patience > 0
                and epoch >= select_start
                and no_improve_epochs >= early_patience
            ):
                early_stopped = True
                break
    finally:
        if old_tqdm is not None:
            ft.tqdm = old_tqdm

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gib = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    else:
        peak_gib = None
    total_sec = time.perf_counter() - total_t0
    avg_sec = (sum(epoch_secs) / len(epoch_secs)) if epoch_secs else None
    flop_est = _estimate_full_training_flops(
        flop_profile,
        epoch_stages,
        train_batches=len(train_dl),
        dev_batches=len(dev_dl),
        test_batches=len(test_dl),
    )

    return {
        "full_training_time": True,
        "train_total_sec": total_sec,
        "epoch_avg_sec": avg_sec,
        "epoch_sec": avg_sec,
        "epoch_secs": epoch_secs,
        "epochs_completed": int(last_completed_epoch),
        "n_epochs_config": int(args.n_epochs),
        "n_epochs_measured": int(n_epochs_to_run),
        "full_training_max_epochs": None if max_epochs is None else int(max_epochs),
        "early_stopped": bool(early_stopped),
        "peak_mem_gib": peak_gib,
        "memory_mode": "full_training_loop",
        "flops_forward": flop_est["flops_avg_epoch"],
        "flops_avg_epoch": flop_est["flops_avg_epoch"],
        "flops_total_training": flop_est["flops_total_training"],
        "flops_backend": None if flop_profile is None else flop_profile.get("backend"),
        "flops_error": flop_est["flops_error"],
        "flops_profile": None if flop_profile is None else flop_profile.get("profiles"),
        "flops_protocol": "avg_epoch_est_train3x_eval1x_stage_weighted",
        "selection_metric": args.selection_metric,
        "best_selection_score": (
            None
            if best_results is None or not math.isfinite(best_selection_score)
            else float(best_selection_score)
        ),
        "last_test_results": _float_metrics(last_test_results),
    }


def measure_worker(
    case: str,
    memory_full_accum: bool,
    *,
    skip_epoch_time: bool = False,
    full_training_time: bool = False,
    full_training_max_epochs: int | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("TQDM_DISABLE", "1")
    if str(_MY_CREATION) not in sys.path:
        sys.path.insert(0, str(_MY_CREATION))

    spec = _CASE_TO_SPEC[case]
    hp_path = spec["hparams"]
    train_py = spec["train_py"]
    build_train_argv = _load_build_train_argv(hp_path)
    with tempfile.TemporaryDirectory(dir=str(_tmp_root())) as tmp:
        argv = build_train_argv(checkpoint_dir=tmp)
        sys.argv = [str(spec["argv0"]), *argv]

        ft = _load_train_module(train_py)
        args = ft.args
        train_dl, dev_dl, test_dl, n_opt = ft.setup_data()
        model, optimizer, scheduler = ft.build_model(n_opt)

        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # CUDA warmup (not counted toward peak memory).
        if torch.cuda.is_available() and not full_training_time:
            optimizer.zero_grad()
            b0 = _microbatch_from_loader(ft, train_dl, ft.DEVICE)
            loss0 = _one_step_loss(ft, model, args, *b0, stage=1)
            loss0.backward()
            optimizer.zero_grad()
            torch.cuda.synchronize()

        # Peak memory: one-step probe by default; full mode measures the actual loop.
        if full_training_time:
            peak_gib = None
            memory_mode = "full_training_loop"
        elif memory_full_accum:
            peak_gib = _peak_memory_accum_window(ft, model, optimizer, train_dl)
            memory_mode = "accum_window"
        else:
            peak_gib = _peak_memory_microbatch(ft, model, optimizer, train_dl)
            memory_mode = "single_microbatch"

        n_batches = len(train_dl)
        epoch_sec: float | None
        if skip_epoch_time or full_training_time:
            epoch_sec = None
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            ft.train_epoch(model, train_dl, optimizer, scheduler, stage=1, epoch_idx=1, ema=None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_sec = time.perf_counter() - t0

        if full_training_time:
            full_flop_profile = _profile_full_training_flops(ft, model, train_dl, dev_dl, test_dl)
            default_flop_profile = full_flop_profile["profiles"]["train_stage1_forward"]
        else:
            full_flop_profile = None
            default_flop_profile = _profile_default_flops(ft, model, train_dl)
        flops = default_flop_profile["flops"]
        flop_backend = default_flop_profile["backend"]
        flop_err = default_flop_profile["error"]

        result = {
            "case": case,
            "dataset": args.dataset,
            "train_batch_size": int(args.train_batch_size),
            "gradient_accumulation_step": int(args.gradient_accumulation_step),
            "max_seq_length": int(args.max_seq_length),
            "params_total": int(total_p),
            "params_trainable": int(train_p),
            "flops_forward": flops,
            "flops_backend": flop_backend,
            "flops_error": flop_err,
            "peak_mem_gib": peak_gib,
            "memory_mode": memory_mode,
            "epoch_sec": epoch_sec,
            "train_batches_per_epoch": int(n_batches),
        }

        del model, optimizer, scheduler, train_dl, dev_dl, test_dl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if full_training_time:
            result.update(
                _time_full_training_run(
                    ft,
                    flop_profile=full_flop_profile,
                    max_epochs=full_training_max_epochs,
                ),
            )

        return result


def _run_worker_subprocess(
    case: str,
    memory_full_accum: bool,
    skip_epoch_time: bool,
    full_training_time: bool,
    full_training_max_epochs: int | None,
) -> dict[str, Any]:
    cmd = [
        _worker_python(),
        "-u",
        str(Path(__file__).resolve()),
        "--measure-worker",
        case,
    ]
    if memory_full_accum:
        cmd.append("--memory-full-accum")
    if skip_epoch_time:
        cmd.append("--skip-epoch-time")
    if full_training_time:
        cmd.append("--full-training-time")
    if full_training_max_epochs is not None:
        cmd.extend(["--full-training-max-epochs", str(int(full_training_max_epochs))])
    proc = subprocess.run(
        cmd,
        cwd=str(_MY_CREATION),
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker {case} failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"worker {case}: empty stdout\nstderr:\n{proc.stderr}")
    return json.loads(lines[-1])


def _md_row(r: dict[str, Any], *, full_mode: bool) -> str:
    def fmt_flops(x: Any) -> str:
        if x is None:
            return "—"
        xf = float(x)
        if xf >= 1e12:
            return f"{xf/1e12:.2f}T"
        if xf >= 1e9:
            return f"{xf/1e9:.2f}G"
        if xf >= 1e6:
            return f"{xf/1e6:.2f}M"
        return f"{xf:.2e}"

    flops_key = "flops_avg_epoch" if full_mode else "flops_forward"
    flops_s = fmt_flops(r.get(flops_key))
    if flops_s == "—" and r.get("flops_error"):
        flops_s = f"— ({r['flops_error'][:80]}…)" if len(str(r.get("flops_error"))) > 80 else f"— ({r.get('flops_error')})"

    mem = r.get("peak_mem_gib")
    mem_s = f"{mem:.3f}" if mem is not None else "—"

    es = r.get("epoch_sec")
    epoch_s = f"{float(es):.2f}" if es is not None else "—"

    if full_mode:
        total_ts = r.get("train_total_sec")
        total_train_s = f"{float(total_ts):.2f}" if total_ts is not None else "—"
        epochs = r.get("epochs_completed")
        epochs_s = str(int(epochs)) if epochs is not None else "—"
        return (
            f"| {r['case']} | {r['params_total']:,} | {r['params_trainable']:,} | {flops_s} | "
            f"{mem_s} | {epoch_s} | {total_train_s} | {epochs_s} | "
            f"{r['train_batches_per_epoch']} | "
            f"{r['train_batch_size']}×{r['gradient_accumulation_step']} | {r['max_seq_length']} |"
        )

    return (
        f"| {r['case']} | {r['params_total']:,} | {r['params_trainable']:,} | {flops_s} | "
        f"{mem_s} | {epoch_s} | {r['train_batches_per_epoch']} | "
        f"{r['train_batch_size']}×{r['gradient_accumulation_step']} | {r['max_seq_length']} |"
    )


def _print_table(rows: list[dict[str, Any]]) -> str:
    full_mode = any(bool(r.get("full_training_time")) for r in rows)
    if full_mode:
        header = (
            "| case | params | trainable | avg_epoch_FLOPs_est | peak_GPU_GiB | "
            "avg_epoch_sec | train_total_sec | epochs_completed | "
            "batches/epoch | B×accum | seq |"
        )
        sep = "|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|"
    else:
        header = (
            "| case | params | trainable | FLOPs (1 fwd, est.) | peak_GPU_GiB | "
            "one_epoch_train_sec | batches/epoch | B×accum | seq |"
        )
        sep = "|---|---:|---:|---|---:|---:|---:|---|---:|"
    lines = [header, sep] + [_md_row(r, full_mode=full_mode) for r in rows]
    text = "\n".join(lines) + "\n"
    print(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--measure-worker",
        metavar="CASE",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--memory-full-accum",
        action="store_true",
        help="Peak memory over a full gradient-accumulation window (no optimizer.step).",
    )
    ap.add_argument(
        "--skip-epoch-time",
        action="store_true",
        help="Skip full train_epoch timing in the worker (faster smoke test).",
    )
    ap.add_argument(
        "--full-training-time",
        action="store_true",
        help=(
            "Run the complete training loop and report total wall time, "
            "completed epochs, and average complete-epoch time."
        ),
    )
    ap.add_argument(
        "--full-training-max-epochs",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--cases",
        default="mosi,mosei,simsv2",
        help="Comma-separated subset of mosi,mosei,simsv2 (driver only).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write Markdown table to this path.",
    )
    args = ap.parse_args()

    if args.measure_worker:
        case = args.measure_worker.strip().lower()
        if case not in _CASE_TO_SPEC:
            print(f"unknown case {case!r}", file=sys.stderr)
            return 2
        result = measure_worker(
            case,
            memory_full_accum=args.memory_full_accum,
            skip_epoch_time=args.skip_epoch_time,
            full_training_time=args.full_training_time,
            full_training_max_epochs=args.full_training_max_epochs,
        )
        print(json.dumps(result), flush=True)
        return 0

    cases = [c.strip().lower() for c in args.cases.split(",") if c.strip()]
    for c in cases:
        if c not in _CASE_TO_SPEC:
            print(f"unknown case in --cases: {c!r}", file=sys.stderr)
            return 2

    rows: list[dict[str, Any]] = []
    for c in cases:
        rows.append(
            _run_worker_subprocess(
                c,
                memory_full_accum=args.memory_full_accum,
                skip_epoch_time=args.skip_epoch_time,
                full_training_time=args.full_training_time,
                full_training_max_epochs=args.full_training_max_epochs,
            ),
        )

    table = _print_table(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        intro = (
            "# Model complexity (paper configs)\n\n"
            "Auto-generated by `measure_fixed_cases.py`. "
            "See [README.md](README.md) for definitions.\n\n"
        )
        args.output.write_text(intro + table, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
