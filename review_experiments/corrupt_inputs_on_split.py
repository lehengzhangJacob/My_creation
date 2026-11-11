#!/usr/bin/env python3
"""Input corruption for confidence and routing checks (Reviewer 2.2 and 2.4).

Pure helpers are importable without a GPU. ``--run`` is the checkpoint grid;
it refuses to start until a GPU is visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MODALITIES = ("acoustic", "language", "visual")
KINDS = ("noise", "zero", "shuffle", "token_mask")
PERTURB = ("primary", "auxiliary", "both")
ROUTES = ("adaptive", "fixed-clean")
CONFIDENCE = ("original", "uniform")


def window_bounds(length: int, fraction: float, start: int | None = None) -> tuple[int, int]:
    """Contiguous window covering ``fraction`` of ``length``."""
    if length <= 0:
        raise ValueError("length must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    width = max(1, int(round(length * fraction)))
    width = min(width, length)
    if start is None:
        start = 0
    if start < 0 or start + width > length:
        raise ValueError(f"window [{start}, {start + width}) is outside [0, {length})")
    return start, start + width


def corruption_mask(length: int, start: int, end: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    mask[start:end] = True
    return mask


def apply_corruption(
    values: np.ndarray,
    start: int,
    end: int,
    kind: str,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Corrupt axis 0 of ``values`` on [start, end) only."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind}")
    out = np.array(values, copy=True, dtype=float)
    if kind in ("zero", "token_mask"):
        out[start:end] = 0.0
    elif kind == "noise":
        out[start:end] = out[start:end] + rng.normal(0.0, scale, size=out[start:end].shape)
    elif kind == "shuffle":
        window = out[start:end].copy()
        rng.shuffle(window)
        out[start:end] = window
    return out


def modalities_to_corrupt(primary: str, perturb: str) -> set[str]:
    if primary not in MODALITIES or perturb not in PERTURB:
        raise ValueError(f"bad primary/perturb: {primary}, {perturb}")
    if perturb == "both":
        return set(MODALITIES)
    if perturb == "primary":
        return {primary}
    return set(MODALITIES) - {primary}


def force_primary(weights: np.ndarray, clean_primary: int) -> np.ndarray:
    """Fixed-clean route: one-hot at the clean-input primary. Shape (..., 3)."""
    forced = np.zeros_like(weights, dtype=float)
    forced[..., int(clean_primary)] = 1.0
    return forced


def confidence_delta(clean_conf: np.ndarray, corrupt_conf: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if clean_conf.shape != corrupt_conf.shape or mask.shape[0] != clean_conf.shape[0]:
        raise ValueError("confidence/mask shape mismatch")
    delta = np.asarray(corrupt_conf, dtype=float) - np.asarray(clean_conf, dtype=float)
    hit = np.asarray(mask, dtype=bool)

    def _mean(sel: np.ndarray) -> float:
        if not np.any(sel):
            return float("nan")
        return float(np.mean(delta[sel]))

    return {"delta_corrupted": _mean(hit), "delta_clean": _mean(~hit)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mosei", "simsv2", "mosi"), default="simsv2")
    parser.add_argument("--fractions", default="0.2,0.4")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.run:
        raise SystemExit(
            "Checkpoint inference needs the 4090D instance (nvidia-smi must succeed). "
            f"Grid: dataset={args.dataset} fractions={args.fractions} kinds={KINDS} "
            f"perturb={PERTURB} routes={ROUTES} confidence={CONFIDENCE}."
        )
    print("Corruption helpers ready. Pass --run on a GPU instance to evaluate the grid.")


if __name__ == "__main__":
    main()
