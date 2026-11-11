#!/usr/bin/env python3
"""Run the reviewer experiment matrix serially on one visible GPU."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from review_common import (
    DEFAULT_MANIFEST,
    HARNESS_DIR,
    MY_CREATION_DIR,
    atomic_write_json,
    checkpoint_dir,
    expected_artifact_paths,
    expand_token,
    experiment_digest,
    load_manifest,
    resolve_my_creation_path,
    select_experiments,
    utc_now,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SELECTOR",
        help=(
            "Run matching IDs/globs or an exact dataset/family/tag. Repeat or "
            "use commas, e.g. --only 'mosei_router_*' --only hcf."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Select the full manifest. Equivalent to the default; "
            "external_running entries are still never launched."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the serial queue without starting subprocesses.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip entries with a matching atomic success marker.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly rerun entries even when success markers exist.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue the serial queue after a failed experiment.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="Exactly one CUDA device identifier (default: 0).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to call frozen launchers.",
    )
    args = parser.parse_args(argv)
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    if args.all and args.only:
        parser.error("--all and --only are mutually exclusive")
    gpu = str(args.cuda_visible_devices).strip()
    if not gpu or "," in gpu or any(char.isspace() for char in gpu):
        parser.error("--cuda-visible-devices must name exactly one GPU")
    args.cuda_visible_devices = gpu
    return args


def build_command(experiment: dict[str, Any], python: str) -> list[str]:
    """Call a frozen launcher and pass control-only overrides."""
    launcher = resolve_my_creation_path(experiment["launcher"])
    overrides = [
        expand_token(token, experiment["id"]) for token in experiment.get("overrides", [])
    ]
    return [
        python,
        "-u",
        str(launcher),
        *overrides,
        "--checkpoint-dir",
        str(checkpoint_dir(experiment)),
    ]


def success_marker_path(experiment: dict[str, Any]) -> Path:
    return HARNESS_DIR / "state" / f"{experiment['id']}.success.json"


def status_path(experiment: dict[str, Any]) -> Path:
    return HARNESS_DIR / "state" / f"{experiment['id']}.status.json"


def _valid_success_marker(experiment: dict[str, Any]) -> tuple[bool, str]:
    marker_path = success_marker_path(experiment)
    if not marker_path.is_file():
        return False, "missing"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable marker: {exc}"
    if marker.get("status") != "success":
        return False, "marker status is not success"
    if marker.get("experiment_id") != experiment["id"]:
        return False, "marker experiment_id differs"
    if marker.get("manifest_entry_sha256") != experiment_digest(experiment):
        return False, "manifest entry changed since success"
    checkpoint_path, predictions_path = expected_artifact_paths(experiment)
    if not checkpoint_path.is_file() or not predictions_path.is_file():
        return False, "success marker artifacts are missing"
    return True, "matching success marker"


def _missing_prerequisites(experiment: dict[str, Any]) -> list[Path]:
    missing: list[Path] = []
    launcher = resolve_my_creation_path(experiment["launcher"])
    if not launcher.is_file():
        missing.append(launcher)
    for raw_path in experiment.get("required_paths", []):
        path = resolve_my_creation_path(raw_path)
        if not path.exists():
            missing.append(path)
    return missing


def _finalize_log(temporary_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_path, final_path)
    try:
        directory_fd = os.open(final_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_one(
    experiment: dict[str, Any],
    *,
    command: list[str],
    cuda_visible_devices: str,
) -> bool:
    exp_id = experiment["id"]
    checkpoint_path, predictions_path = expected_artifact_paths(experiment)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = HARNESS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (HARNESS_DIR / "state").mkdir(parents=True, exist_ok=True)

    final_log = logs_dir / f"{exp_id}.log"
    temporary_log = logs_dir / f".{exp_id}.log.{os.getpid()}.running"
    started_at = utc_now()
    start_monotonic = time.monotonic()
    digest = experiment_digest(experiment)
    base_status = {
        "schema_version": 1,
        "experiment_id": exp_id,
        "manifest_entry_sha256": digest,
        "command": command,
        "cwd": str(MY_CREATION_DIR),
        "cuda_visible_devices": cuda_visible_devices,
        "started_at": started_at,
        "log": str(final_log),
        "temporary_log": str(temporary_log),
    }
    atomic_write_json(status_path(experiment), {**base_status, "status": "starting"})

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TQDM_DISABLE", "1")
    env["PYTHONUNBUFFERED"] = "1"

    process: subprocess.Popen[Any] | None = None
    return_code: int | None = None
    failure_reason: str | None = None
    with temporary_log.open("wb") as log_handle:
        header = (
            f"# experiment_id: {exp_id}\n"
            f"# started_at: {started_at}\n"
            f"# cwd: {MY_CREATION_DIR}\n"
            f"# CUDA_VISIBLE_DEVICES: {cuda_visible_devices}\n"
            f"# command: {shlex.join(command)}\n\n"
        )
        log_handle.write(header.encode("utf-8"))
        log_handle.flush()
        os.fsync(log_handle.fileno())
        try:
            process = subprocess.Popen(
                command,
                cwd=str(MY_CREATION_DIR),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            atomic_write_json(
                status_path(experiment),
                {**base_status, "status": "running", "pid": process.pid},
            )
            return_code = process.wait()
            if return_code != 0:
                failure_reason = f"subprocess exited with code {return_code}"
        except BaseException as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            if process is not None:
                _terminate_process(process)
                return_code = process.returncode
            log_handle.write(f"\n# launcher exception: {failure_reason}\n".encode("utf-8"))
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                log_handle.flush()
                os.fsync(log_handle.fileno())
                ended_at = utc_now()
                elapsed = time.monotonic() - start_monotonic
                atomic_write_json(
                    status_path(experiment),
                    {
                        **base_status,
                        "status": "interrupted",
                        "pid": process.pid if process else None,
                        "return_code": return_code,
                        "ended_at": ended_at,
                        "elapsed_seconds": elapsed,
                        "error": failure_reason,
                    },
                )
                _finalize_log(temporary_log, final_log)
                raise
        finally:
            log_handle.flush()
            os.fsync(log_handle.fileno())

    _finalize_log(temporary_log, final_log)
    if failure_reason is None:
        missing_outputs = [
            str(path)
            for path in (checkpoint_path, predictions_path)
            if not path.is_file()
        ]
        if missing_outputs:
            failure_reason = "successful subprocess did not create: " + ", ".join(
                missing_outputs
            )

    ended_at = utc_now()
    elapsed = time.monotonic() - start_monotonic
    if failure_reason is not None:
        atomic_write_json(
            status_path(experiment),
            {
                **base_status,
                "status": "failed",
                "pid": process.pid if process else None,
                "return_code": return_code,
                "ended_at": ended_at,
                "elapsed_seconds": elapsed,
                "error": failure_reason,
            },
        )
        print(f"FAILED {exp_id}: {failure_reason}", file=sys.stderr)
        return False

    success = {
        **base_status,
        "status": "success",
        "pid": process.pid if process else None,
        "return_code": return_code,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint_path),
        "predictions": str(predictions_path),
    }
    atomic_write_json(status_path(experiment), success)
    atomic_write_json(success_marker_path(experiment), success)
    print(f"DONE   {exp_id} ({elapsed:.1f}s)")
    return True


def _acquire_serial_lock() -> Any:
    state_dir = HARNESS_DIR / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (state_dir / "launcher.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            "another review launcher holds the serial lock; refusing parallel training"
        ) from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
    lock_handle.flush()
    os.fsync(lock_handle.fileno())
    return lock_handle


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    selected = select_experiments(manifest["experiments"], args.only)

    train_entries = [exp for exp in selected if exp["action"] == "train"]
    reuse_entries = [exp for exp in selected if exp["action"] == "reuse"]
    external_entries = [
        exp for exp in selected if exp["action"] == "external_running"
    ]
    print(
        f"Selected {len(selected)} entries: {len(train_entries)} train, "
        f"{len(reuse_entries)} reuse, {len(external_entries)} external_running"
    )
    for exp in reuse_entries:
        source = exp.get("source", {})
        source_text = ", ".join(f"{key}={value}" for key, value in source.items())
        print(f"REUSE  {exp['id']} <- {source_text}")
    for exp in external_entries:
        print(
            f"EXTERNAL {exp['id']} [NEVER LAUNCH] "
            f"{exp.get('reason', 'managed outside this harness')}"
        )

    missing_by_id = {
        exp["id"]: _missing_prerequisites(exp)
        for exp in train_entries
        if _missing_prerequisites(exp)
    }

    commands = {
        exp["id"]: build_command(exp, args.python)
        for exp in train_entries
    }
    if args.dry_run:
        print(
            f"DRY RUN: serial execution, CUDA_VISIBLE_DEVICES="
            f"{args.cuda_visible_devices}"
        )
        for index, exp in enumerate(train_entries, start=1):
            missing_note = ""
            if exp["id"] in missing_by_id:
                missing_note = "  [MISSING: " + ", ".join(
                    str(path) for path in missing_by_id[exp["id"]]
                ) + "]"
            print(
                f"{index:02d}/{len(train_entries):02d} {exp['id']}{missing_note}\n"
                f"    {shlex.join(commands[exp['id']])}"
            )
        if missing_by_id:
            print(
                "DRY RUN WARNING: missing prerequisites are listed above; "
                "real execution will refuse to start.",
                file=sys.stderr,
            )
        return 0

    if not train_entries:
        print("No runnable training entries selected.")
        return 0

    if missing_by_id:
        details = "; ".join(
            f"{exp_id}: {', '.join(str(path) for path in paths)}"
            for exp_id, paths in missing_by_id.items()
        )
        raise FileNotFoundError(f"missing training prerequisites: {details}")

    lock_handle = _acquire_serial_lock()
    failures = 0
    try:
        for index, exp in enumerate(train_entries, start=1):
            marker_valid, marker_reason = _valid_success_marker(exp)
            if marker_valid and args.resume:
                print(
                    f"SKIP   {index:02d}/{len(train_entries):02d} {exp['id']} "
                    f"({marker_reason})"
                )
                continue
            if marker_valid and not args.force:
                raise RuntimeError(
                    f"{exp['id']} already succeeded; use --resume to skip or "
                    "--force to rerun"
                )
            marker_exists = success_marker_path(exp).exists()
            if marker_exists and not marker_valid and args.resume:
                raise RuntimeError(
                    f"{exp['id']} has a stale/invalid success marker "
                    f"({marker_reason}); inspect it or use --force"
                )

            print(
                f"START  {index:02d}/{len(train_entries):02d} {exp['id']} "
                f"on CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}"
            )
            ok = run_one(
                exp,
                command=commands[exp["id"]],
                cuda_visible_devices=args.cuda_visible_devices,
            )
            if not ok:
                failures += 1
                if not args.keep_going:
                    break
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
