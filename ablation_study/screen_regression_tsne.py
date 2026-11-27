"""CPU silhouette screen of regression full checkpoints.

Prints 3-class and positive/negative (neutral dropped) t-SNE silhouettes.
Does not touch the GPU.
"""
from __future__ import annotations

import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_MY = _HERE.parent
sys.path.insert(0, str(_HERE))
os.chdir(_MY)

import tsne_sota3090_more as T  # noqa: E402

SEARCH = _MY / "sota_search" / "runs"
OUT = _HERE / "runs" / "sota3090_tsne" / "regression_screen"
CASES = [
    ("mosi_bert", "878"),
    ("mosi_bert", "2132"),
    ("mosi_bert", "2133"),
    ("mosi_bert", "4753"),
    ("mosei_bert", "93"),
    ("mosei_bert", "199"),
    ("mosei_bert", "163"),
    ("mosei_bert", "183"),
    ("mosei_bert", "201"),
]


def _ckpt(row: str, trial: str) -> Path | None:
    root = SEARCH / row
    candidates = [
        root / f"trial_{trial}" / "checkpoints",
        root / f"primary_replay_{trial}" / "checkpoints",
    ]
    for folder in candidates:
        hits = sorted(folder.glob("infogate_*best.pt"))
        if hits:
            return hits[0]
    return None


def _cmd(row: str, trial: str) -> Path | None:
    path = SEARCH / row / f"trial_{trial}" / "cmd.json"
    return path if path.is_file() else None


def _argv(cmd_json: Path, ckpt_dir: Path) -> list[str]:
    cmd = json.loads(cmd_json.read_text())["cmd"]
    argv = list(cmd[3:])
    out = []
    skip = False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in ("--checkpoint_dir", "--train_batch_size"):
            skip = True
            continue
        out.append(tok)
    out += ["--checkpoint_dir", str(ckpt_dir), "--train_batch_size", "8"]
    return out


def _posneg(y: np.ndarray):
    b = np.clip(np.rint(y.astype(np.float64)), -3, 3).astype(np.int32)
    mask = b != 0
    labels = (b[mask] > 0).astype(np.int32)
    return mask, labels


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'trial':<22} {'n':>5} {'sil3':>7} {'npos':>6} {'sil_pn':>7}", flush=True)
    best = None
    for row, trial in CASES:
        ckpt = _ckpt(row, trial)
        cmd = _cmd(row, trial)
        key = f"{row}_{trial}"
        if ckpt is None or cmd is None:
            print(f"{key:<22} MISSING ckpt={ckpt is not None} cmd={cmd is not None}", flush=True)
            continue
        try:
            mod = T._load_module("train.py", _argv(cmd, ckpt.parent), "_screen_reg")
            H, y = T._collect(mod, ckpt, "regression")
        except Exception as exc:
            print(f"{key:<22} FAIL {type(exc).__name__}: {exc}", flush=True)
            continue
        labels3 = T._labels(y, "acc7")
        emb3 = T._tsne(H)
        sil3 = T._silhouette(emb3, labels3)
        mask, labels = _posneg(y)
        emb = T._tsne(H[mask])
        sil = T._silhouette(emb, labels)
        npos = int(labels.sum())
        print(f"{key:<22} {len(y):5d} {sil3:7.3f} {npos:6d} {sil:7.3f}", flush=True)
        if best is None or sil > best[0]:
            best = (sil, key, emb, labels, npos, len(y) - int(mask.sum()) )
        np.savez(OUT / f"{key}_posneg.npz", emb=emb, labels=labels)
    if best is not None:
        sil, key, emb, labels, npos, nneu = best
        print(f"BEST_POSNEG {key} silhouette={sil:.3f} npos={npos} neutral_dropped={nneu}", flush=True)
    print("SCREEN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
