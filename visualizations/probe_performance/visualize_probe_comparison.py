import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.parent
NATURAL_PROBE_PATH = SCRIPT_DIR / "notebookV3" / "results" / "result_both_linear.csv"
NON_NATURAL_PROBE_PATH = SCRIPT_DIR / "notebookV3" / "results" / "result_both_linear_wrong.csv"
OUTPUT_PATH = Path(__file__).parent / "probe_comparison.png"

TOP_K = 10
FIGSIZE = (16, 10)
MARKER_SIZE = 13
LINE_WIDTH = 5

LABEL_FONTSIZE = 24
TITLE_FONTSIZE = 34
AXIS_LABEL_FONTSIZE = 32
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 28

natural_df = pd.read_csv(NATURAL_PROBE_PATH)
non_natural_df = pd.read_csv(NON_NATURAL_PROBE_PATH)

natural_top = natural_df.nlargest(TOP_K, "val_acc").reset_index(drop=True)
non_natural_top = non_natural_df.nlargest(TOP_K, "val_acc").reset_index(drop=True)

fig, ax = plt.subplots(figsize=FIGSIZE)

ranks = np.arange(1, TOP_K + 1)

ax.plot(
    ranks,
    natural_top["val_acc"],
    marker="o",
    markersize=MARKER_SIZE,
    linewidth=LINE_WIDTH,
    label="Natural Probe",
    color="green",
    linestyle="-",
)
ax.plot(
    ranks,
    non_natural_top["val_acc"],
    marker="s",
    markersize=MARKER_SIZE,
    linewidth=LINE_WIDTH,
    label="Non-Natural Probe",
    color="red",
    linestyle="-",
)

for i in range(TOP_K):
    natural_offset = 0.03 if i % 2 == 0 else 0.05
    ax.text(
        ranks[i],
        natural_top["val_acc"].iloc[i] + natural_offset,
        f"L{natural_top['layer'].iloc[i]}H{natural_top['head'].iloc[i]}",
        fontsize=LABEL_FONTSIZE,
        ha="center",
        color="darkgreen",
        weight="bold",
    )

    non_natural_offset = 0.06 if i % 2 == 0 else 0.08
    ax.text(
        ranks[i],
        non_natural_top["val_acc"].iloc[i] - non_natural_offset,
        f"L{non_natural_top['layer'].iloc[i]}H{non_natural_top['head'].iloc[i]}",
        fontsize=LABEL_FONTSIZE,
        ha="center",
        color="darkred",
        weight="bold",
    )

ax.set_xlabel("Rank (Best to Worst)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
ax.set_ylabel("Extraction Accuracy", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
ax.set_title(
    f"Top {TOP_K} Attention Head Probes: Natural vs Non-Natural\n(One probe trained per attention head)",
    fontsize=TITLE_FONTSIZE,
    fontweight="bold",
    pad=25,
)
ax.set_xticks(ranks)
ax.set_ylim(0.2, 1.05)
ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
print(f"\n✓ Figure saved to: {OUTPUT_PATH}")
plt.show()
