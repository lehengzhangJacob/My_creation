import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

methods = {
    "2": ["Self-MM","FDMER", "MISA" ,"UCMIB-PNS", "AtCAF", "SuCI", "DHMD","RITHM"],
    "3": ["MAG-XLNet", "HKT", "MuLOT", "MIL", "MCL", "MGCL", "DHMD", "PRISM"]
}

data = {
    "2": [68.77,70.55,70.61, 72.13,  72.13, 70.92, 71.30,  75.35],
    "3": [74.72, 76.47, 76.82, 76.30, 77.94, 77.94, 78.50, 79.41]
}

datasets = ["2", "3"]

colors = [
    "#d9ddd3",  # 浅灰绿（FedAvg）
    "#c8d6b9",  # 淡绿（FedNOVA）
    "#a8c97f",  # 草绿（FedProto）
    "#8fd0c1",  # 青绿（MOON）
    "#5aa6a5",  # 蓝绿（AdaFGL）
    "#4e9aa6",  # 青蓝（FedGTA）
    "#2e7d84",  # 深青（FedType）
    "#1f5f7a"   # 深蓝（Ours）
]

y_config = {
    "2": {"ylim": (66, 78), "ylabel": "Accuracy  (%)"},
    "3": {"ylim": (70, 82), "ylabel": "Accuracy  (%)"}
}

xlabel_config = {
    "2": "UR-FUNNY v2",
    "3": "MUStARD"
}

fig, axes = plt.subplots(1, 2, figsize=(8.8, 4))

for idx, dataset in enumerate(datasets):
    ax = axes[idx]

    values = data[dataset]
    labels = methods[dataset]
    x = np.arange(len(values))

    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

    bars = ax.bar(
        x,
        values,
        width=0.6,
        color=colors,
        edgecolor="black",
        linewidth=1,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, rotation=30, ha="right")

    ax.set_xlabel(xlabel_config[dataset], fontsize=14, fontweight="bold", labelpad=10)
    ax.xaxis.set_label_coords(0.5, -0.24)
    ax.set_ylabel(y_config[dataset]["ylabel"], fontsize=13, fontweight="bold")

    ax.set_ylim(y_config[dataset]["ylim"])
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
            zorder=5,
        )
    ax.tick_params(axis="y", labelsize=13)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

plt.subplots_adjust(
    left=0.10,
    right=0.98,
    bottom=0.22,
    top=0.88,
    wspace=0.28
)


plt.savefig("mhd_msd.pdf", dpi=300, bbox_inches="tight")
plt.show()