"""t-SNE the two leftover classification ablation sets that are not paper SOTA."""
import os
import sys
from pathlib import Path

import numpy as np

ABL = Path("/home/gmn/danger/codes/PRISM2/My_creation/ablation_study")
MY = ABL.parent
PREV = Path("/home/gmn/danger/codes/PRISM2/fixed_experiment_previous/runs")
sys.path.insert(0, str(ABL))
os.chdir(MY)

import tsne_sota3090_more as T  # noqa: E402
from mustard_s2_local_trial134_hparams import build_train_argv as mustard_argv  # noqa: E402
from ur_funny_trial162_hparams import build_train_argv as urfunny_argv  # noqa: E402

JOBS = (
    {
        "key": "mustard_trial134",
        "title": "MUStARD trial 134",
        "classes": ("non-sarcasm", "sarcasm"),
        "builder": mustard_argv,
        "ckpts": {
            "none": PREV / "mustard_s2_local_trial134/checkpoints/infogate_mustard_best.pt",
            "no_ib": ABL / "runs/mustard_s2_local_trial134_no_ib/checkpoints/infogate_mustard_best.pt",
            "no_mselector": ABL / "runs/mustard_s2_local_trial134_no_mselector/checkpoints/infogate_mustard_best.pt",
            "no_infogate": ABL / "runs/mustard_s2_local_trial134_no_infogate/checkpoints/infogate_mustard_best.pt",
        },
    },
    {
        "key": "ur_funny_trial162",
        "title": "UR-FUNNY v2 trial 162",
        "classes": ("not funny", "funny"),
        "builder": urfunny_argv,
        "ckpts": {
            "none": PREV / "ur_funny_trial162/checkpoints/infogate_ur_funny_best.pt",
            "no_ib": ABL / "runs/ur_funny_trial162_no_ib/checkpoints/infogate_ur_funny_best.pt",
            "no_mselector": ABL / "runs/ur_funny_trial162_no_mselector/checkpoints/infogate_ur_funny_best.pt",
            "no_infogate": ABL / "runs/ur_funny_trial162_no_infogate/checkpoints/infogate_ur_funny_best.pt",
        },
    },
)
TITLES = {
    "none": "PRISM (full)",
    "no_ib": "w/o VTB",
    "no_mselector": "w/o DPR",
    "no_infogate": "w/o InfoGate",
}


def main() -> int:
    T._style()
    for job in JOBS:
        fam = {
            "key": job["key"],
            "title": job["title"],
            "kind": "binary",
            "loader": "classify",
            "classes": job["classes"],
        }
        panels = []
        for ablation, title in T.VARIANTS:
            ckpt = job["ckpts"][ablation]
            print(f"=== {job['key']} {ablation} {ckpt} ===", flush=True)
            if not ckpt.is_file():
                raise FileNotFoundError(ckpt)
            argv = job["builder"](
                checkpoint_dir=str(ckpt.parent),
                ablation="none" if ablation == "none" else ablation,
            )
            mod = T._load_module("train_classify.py", argv, "_tsne_extra_train")
            H, y = T._collect(mod, ckpt, "classify")
            labels = T._labels(y, "binary")
            emb = T._tsne(H)
            sil = T._silhouette(emb, labels)
            counts = {job["classes"][i]: int(np.sum(labels == i)) for i in (0, 1)}
            print(f"  n={len(y)} {counts} silhouette={sil:.3f}", flush=True)
            panels.append((title, emb, labels, sil))
        full = panels[0][3]
        rest = ", ".join(f"{title} {sil:.3f}" for title, _, _, sil in panels[1:])
        print(f"COMPARE {job['key']} full {full:.3f} vs {rest}", flush=True)
        T._plot(fam, panels)
    print("TSNE_EXTRA_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
