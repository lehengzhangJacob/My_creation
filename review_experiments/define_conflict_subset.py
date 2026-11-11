#!/usr/bin/env python3
"""Conflict subsets defined without PRISM outputs (Reviewer 2.3).

CH-SIMS v2 uses unimodal labels when the pickle contains them. MOSI/MOSEI
use sign disagreement among three least-squares unimodal probes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REASON = {
    0: "none",
    1: "text_vs_audio",
    2: "text_vs_visual",
    3: "text_vs_audio_and_visual",
}


def _sign(values: np.ndarray, min_abs: float) -> np.ndarray:
    out = np.zeros(np.shape(values), dtype=int)
    out[values >= min_abs] = 1
    out[values <= -min_abs] = -1
    return out


def conflict_from_unimodal_labels(
    text: np.ndarray,
    audio: np.ndarray,
    visual: np.ndarray,
    min_abs: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Opposite-sign conflicts. Near-zero labels are not conflicts.

    reason: 0 none, 1 text-vs-audio, 2 text-vs-visual, 3 both.
    """
    text = np.asarray(text, dtype=float).reshape(-1)
    audio = np.asarray(audio, dtype=float).reshape(-1)
    visual = np.asarray(visual, dtype=float).reshape(-1)
    if not (text.shape == audio.shape == visual.shape):
        raise ValueError("unimodal label lengths differ")
    st, sa, sv = _sign(text, min_abs), _sign(audio, min_abs), _sign(visual, min_abs)
    vs_audio = (st * sa) < 0
    vs_visual = (st * sv) < 0
    reason = vs_audio.astype(int) + 2 * vs_visual.astype(int)
    return reason > 0, reason


def conflict_from_probe_signs(
    sign_t: np.ndarray,
    sign_a: np.ndarray,
    sign_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sign_t = np.asarray(sign_t, dtype=int).reshape(-1)
    sign_a = np.asarray(sign_a, dtype=int).reshape(-1)
    sign_v = np.asarray(sign_v, dtype=int).reshape(-1)
    vs_audio = (sign_t * sign_a) < 0
    vs_visual = (sign_t * sign_v) < 0
    reason = vs_audio.astype(int) + 2 * vs_visual.astype(int)
    return reason > 0, reason


def fit_unimodal_probe(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Least-squares weights with a bias column. Features are (N, D)."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature/label count mismatch")
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    return weights


def predict_probe(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return design @ np.asarray(weights, dtype=float)


def write_subset(path: Path, is_conflict: np.ndarray, reason: np.ndarray, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("global_index", "is_conflict", "reason", "source")
        )
        writer.writeheader()
        for index, (flag, code) in enumerate(zip(is_conflict, reason)):
            writer.writerow({
                "global_index": index,
                "is_conflict": int(bool(flag)),
                "reason": REASON[int(code)],
                "source": source,
            })
    print(f"Wrote {path} ({int(np.sum(is_conflict))}/{len(is_conflict)} conflicts)")


def _load_simsv2_unimodal(pkl_path: Path):
    import pickle
    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)
    split = data["test"]
    needed = ("regression_labels_T", "regression_labels_A", "regression_labels_V")
    missing = [key for key in needed if key not in split]
    if missing:
        raise KeyError(
            "simsv2.pkl has no "
            + ", ".join(missing)
            + ". Align the official CH-SIMS v2 meta.csv by clip id before retrying."
        )
    return tuple(np.asarray(split[key], dtype=float) for key in needed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("simsv2", "mosei", "mosi"), required=True)
    parser.add_argument("--pkl", type=Path, default=None)
    parser.add_argument("--min-abs", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (HERE / "results" / f"conflict_subset_{args.dataset}.csv")
    if args.dataset != "simsv2":
        raise SystemExit(
            "MOSI/MOSEI subsets need unimodal probes. Fit them with "
            "fit_unimodal_probe on a GPU instance, then write_subset."
        )
    pkl = args.pkl or (HERE.parent / "datasets" / "simsv2.pkl")
    text, audio, visual = _load_simsv2_unimodal(pkl)
    flags, reason = conflict_from_unimodal_labels(text, audio, visual, args.min_abs)
    write_subset(out, flags, reason, source="simsv2_unimodal_labels")


if __name__ == "__main__":
    main()
