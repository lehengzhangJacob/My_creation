"""t-SNE a classification trial from its search cmd.json and local checkpoints.

Full checkpoint is ``<root>/checkpoints/infogate_*best.pt``.
Ablations are ``<root>_no_ib``, ``<root>_no_mselector``, ``<root>_no_infogate``.
``--full-only`` draws just the full model and prints its silhouette.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_MY = _HERE.parent
sys.path.insert(0, str(_HERE))
os.chdir(_MY)

import tsne_sota3090_more as T  # noqa: E402

ABLATIONS = ("no_ib", "no_mselector", "no_infogate")


def _argv(cmd_json: Path, ablation: str, ckpt_dir: Path) -> list[str]:
    cmd = json.loads(cmd_json.read_text())["cmd"]
    argv = list(cmd[3:])
    out = []
    skip = False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in ("--checkpoint_dir", "--ablation"):
            skip = True
            continue
        out.append(tok)
    out += ["--checkpoint_dir", str(ckpt_dir)]
    if ablation != "none":
        out += ["--ablation", ablation]
    return out


def _best_pt(directory: Path) -> Path:
    hits = sorted(directory.glob("infogate_*best.pt"))
    if not hits:
        raise FileNotFoundError(directory)
    return hits[0]


def run(cmd_json: Path, root: Path, key: str, title: str, classes: tuple[str, str], full_only: bool, regression: bool) -> None:
    T._style()
    fam = {
        "key": key,
        "title": title,
        "kind": "acc7" if regression else "binary",
        "loader": "regression" if regression else "classify",
        "classes": classes,
    }
    variants = [("none", "PRISM (full)")]
    if not full_only:
        variants += [
            ("no_ib", "w/o VTB"),
            ("no_mselector", "w/o DPR"),
            ("no_infogate", "w/o InfoGate"),
        ]
    script = "train.py" if regression else "train_classify.py"
    panels = []
    for ablation, panel_title in variants:
        folder = root if ablation == "none" else Path(str(root) + f"_{ablation}")
        ckpt = _best_pt(folder / "checkpoints")
        print(f"=== {key} {ablation} {ckpt} ===", flush=True)
        mod = T._load_module(script, _argv(cmd_json, ablation, ckpt.parent), "_tsne_cmd_train")
        H, y = T._collect(mod, ckpt, fam["loader"])
        labels = T._labels(y, "acc7" if regression else "binary")
        emb = T._tsne(H)
        sil = T._silhouette(emb, labels)
        if regression:
            counts = {name: int(np.sum(labels == name)) for name in ("negative", "neutral", "positive")}
        else:
            counts = {classes[i]: int(np.sum(labels == i)) for i in (0, 1)}
        print(f"  n={len(y)} {counts} silhouette={sil:.3f}", flush=True)
        panels.append((panel_title, emb, labels, sil))
    if full_only:
        print(f"FULL_ONLY {key} silhouette={panels[0][3]:.3f}", flush=True)
        return
    rest = ", ".join(f"{name} {sil:.3f}" for name, _, _, sil in panels[1:])
    print(f"COMPARE {key} full {panels[0][3]:.3f} vs {rest}", flush=True)
    T._plot(fam, panels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--full-only", action="store_true")
    ap.add_argument("--regression", action="store_true")
    args = ap.parse_args()
    run(args.cmd, args.root, args.key, args.title, tuple(args.classes), args.full_only, args.regression)
    print("TSNE_CMD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
