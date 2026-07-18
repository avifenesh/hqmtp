#!/usr/bin/env python3
"""Two non-data figures: head cost anatomy (Fig 2) and the cut taxonomy grid."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})

FID = "#2166ac"
FID_L = "#d1e5f0"
FUN = "#b2182b"
FUN_L = "#fddbc7"
GREY = "#4d4d4d"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


# ---------------- Fig 2: head cost anatomy ----------------
# bytes read per drafted token (9B): output projection + transformer block
rows = [
    ("reference head\n(248,320-row projection, Q8)", 1.08, 0.26),
    ("trim to 32k rows (Q8)", 0.14, 0.26),
    ("trim + NVFP4 (the recipe)", 0.07, 0.14),
]
fig, ax = plt.subplots(figsize=(4.6, 1.9))
y = [2, 1, 0]
for yi, (label, proj, blk) in zip(y, rows):
    ax.barh(yi, proj, 0.55, color="#8073ac", label="output projection" if yi == 2 else None)
    ax.barh(yi, blk, 0.55, left=proj, color="#b2abd2", label="transformer block" if yi == 2 else None)
    ax.text(proj + blk + 0.02, yi, f"{proj + blk:.2f} GB", va="center", fontsize=8)
ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
ax.set_xlabel("weights read per drafted token (GB)")
ax.set_xlim(0, 1.55)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "f0_cost_anatomy")

# ---------------- Fig: cut taxonomy grid ----------------
fig, ax = plt.subplots(figsize=(5.2, 2.6))
ax.axis("off")


def box(x, y, w, h, fc, ec, txt, fs=8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            weight=weight)


ax.text(0.5, 0.97, "What does the cut remove?", ha="center", fontsize=9.5, weight="bold")

box(0.02, 0.60, 0.46, 0.26, FID_L, FID,
    "FIDELITY CUTS\nuniform precision\nof every weight", 8, "bold")
box(0.52, 0.60, 0.46, 0.26, FUN_L, FUN,
    "FUNCTION CUTS\nselected structure,\nor a selected refit", 8, "bold")

box(0.02, 0.36, 0.46, 0.18, "white", FID,
    "Q8   Q4_0   NVFP4", 8)
box(0.52, 0.36, 0.46, 0.18, "white", FUN,
    "channel pruning\nlow-rank (SVD)   distillation", 7.5)

box(0.02, 0.04, 0.46, 0.26, FID_L, FID,
    "shift tax $\\leq$ 1.9 pts\nharm / disagreement $\\approx$ 0.14\ntie-flips, near-equal tokens", 7)
box(0.52, 0.04, 0.46, 0.26, FUN_L, FUN,
    "shift tax 10 to 19 pts\nharm / disagreement 0.28 to 0.54\nrare-content machinery deleted", 7)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
save(fig, "f6_taxonomy")
