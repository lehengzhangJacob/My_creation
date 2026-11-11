#!/usr/bin/env python3
"""Shared manifest and atomic-I/O helpers for the review harness."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HARNESS_DIR = Path(__file__).resolve().parent
MY_CREATION_DIR = HARNESS_DIR.parent
DEFAULT_MANIFEST = HARNESS_DIR / "manifest.json"

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_ACTIONS = {"train", "reuse", "external_running"}
_ALLOWED_OVERRIDE_FLAGS = {
    "--seed": 1,
    "--text-backbone": 1,
    "--model-variant": 1,
    "--model": 1,
    "--confidence-mode": 1,
    "--routing-mode": 1,
    "--disable-l-rib": 0,
    "--disable-hcf": 0,
}


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _validate_overrides(experiment_id: str, tokens: list[str]) -> None:
    i = 0
    while i < len(tokens):
        flag = tokens[i]
        if flag not in _ALLOWED_OVERRIDE_FLAGS:
            raise ValueError(
                f"{experiment_id}: unsupported override {flag!r}; the review "
                "manifest may contain controls only, not copied hyperparameters"
            )
        arity = _ALLOWED_OVERRIDE_FLAGS[flag]
        if i + arity >= len(tokens):
            raise ValueError(f"{experiment_id}: {flag} is missing its value")
        if arity and tokens[i + 1].startswith("--"):
            raise ValueError(f"{experiment_id}: {flag} has an invalid value")
        i += arity + 1


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the small, intentionally strict manifest schema."""
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("manifest experiments must be a non-empty list")

    seen: set[str] = set()
    for exp in experiments:
        if not isinstance(exp, dict):
            raise ValueError("each manifest experiment must be an object")
        exp_id = exp.get("id")
        if not isinstance(exp_id, str) or not _ID_RE.fullmatch(exp_id):
            raise ValueError(f"invalid experiment id: {exp_id!r}")
        if exp_id in seen:
            raise ValueError(f"duplicate experiment id: {exp_id}")
        seen.add(exp_id)

        action = exp.get("action")
        if action not in _ACTIONS:
            raise ValueError(
                f"{exp_id}: action must be train, reuse, or external_running"
            )
        if exp.get("task") not in {"regression", "binary"}:
            raise ValueError(f"{exp_id}: task must be regression or binary")

        if action == "train":
            if not isinstance(exp.get("launcher"), str):
                raise ValueError(f"{exp_id}: train entry requires launcher")
            overrides = exp.get("overrides", [])
            if not isinstance(overrides, list) or not all(
                isinstance(token, str) for token in overrides
            ):
                raise ValueError(f"{exp_id}: overrides must be a string list")
            _validate_overrides(exp_id, overrides)
            seed = int(exp.get("seed", -1))
            allowed = {128}
            if exp.get("dataset") == "simsv2":
                allowed = {128, 42, 2024}
            if seed not in allowed:
                raise ValueError(
                    f"{exp_id}: seed must be one of {sorted(allowed)}"
                )
            if "--seed" in overrides:
                override_seed = int(overrides[overrides.index("--seed") + 1])
                if override_seed != seed:
                    raise ValueError(f"{exp_id}: --seed override must match seed")
            if (
                exp.get("dataset") in {"mosi", "mosei"}
                and "--text-backbone" in overrides
                and overrides[overrides.index("--text-backbone") + 1] == "bert"
            ):
                raise ValueError(
                    f"{exp_id}: MOSI/MOSEI PRISM+BERT is externally managed "
                    "and must never be runnable from this harness"
                )
        else:
            if "launcher" in exp or "overrides" in exp:
                raise ValueError(
                    f"{exp_id}: {action} entries must not define a launcher or overrides"
                )
            source = exp.get("source")
            if not isinstance(source, dict) or not source:
                raise ValueError(f"{exp_id}: {action} entry requires source fields")
    return manifest


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return validate_manifest(manifest)


def split_selectors(values: Iterable[str] | None) -> list[str]:
    selectors: list[str] = []
    for value in values or []:
        selectors.extend(part.strip() for part in value.split(",") if part.strip())
    return selectors


def select_experiments(
    experiments: Iterable[dict[str, Any]], selectors: Iterable[str] | None
) -> list[dict[str, Any]]:
    """Select by id glob or exact dataset/family/condition/tag."""
    requested = split_selectors(selectors)
    selected: list[dict[str, Any]] = []
    for exp in experiments:
        if not requested:
            selected.append(exp)
            continue
        labels = {
            exp["id"],
            str(exp.get("dataset", "")),
            str(exp.get("family", "")),
            str(exp.get("condition", "")),
            *(str(tag) for tag in exp.get("tags", [])),
        }
        if any(
            selector in labels
            or fnmatch.fnmatchcase(exp["id"], selector)
            or any(fnmatch.fnmatchcase(label, selector) for label in labels)
            for selector in requested
        ):
            selected.append(exp)
    if requested and not selected:
        raise ValueError(f"--only matched no experiments: {', '.join(requested)}")
    return selected


def expand_token(token: str, experiment_id: str) -> str:
    return token.format(
        my_creation=str(MY_CREATION_DIR),
        harness=str(HARNESS_DIR),
        experiment_id=experiment_id,
    )


def run_dir(experiment: dict[str, Any]) -> Path:
    return HARNESS_DIR / "runs" / experiment["id"]


def checkpoint_dir(experiment: dict[str, Any]) -> Path:
    return run_dir(experiment) / "checkpoints"


def expected_artifact_paths(experiment: dict[str, Any]) -> tuple[Path, Path]:
    ckpt_dir = checkpoint_dir(experiment)
    checkpoint_name = experiment.get(
        "expected_checkpoint", f"infogate_{experiment['dataset']}_best.pt"
    )
    return ckpt_dir / checkpoint_name, ckpt_dir / "best_test_predictions.csv"


def resolve_my_creation_path(raw_path: str) -> Path:
    path = Path(expand_token(raw_path, "unused")).expanduser()
    return path.resolve() if path.is_absolute() else (MY_CREATION_DIR / path).resolve()


def experiment_digest(experiment: dict[str, Any]) -> str:
    payload = json.dumps(
        experiment, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _atomic_replace_bytes(path, payload)


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_replace_bytes(path, text.encode("utf-8"))
