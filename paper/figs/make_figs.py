#!/usr/bin/env python3
"""Generate the paper's five data figures from results/ JSONs and recorded aggregates.

F1 headline speedup bars    F2 retention vs shift distance   F3 per-bin damage
F4 audit scatter            F5 hot-ratio histogram
Outputs PDF+PNG per figure into paper/figs/.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")

FID = "#2166ac"   # fidelity cuts
FID2 = "#67a9cf"
FUN = "#b2182b"   # function cuts
FUN2 = "#ef8a62"
REF = "#4d4d4d"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

ARMS = ["q8", "q4", "nvfp4", "prune25", "prune50", "svd768", "svdw768"]
LABEL = {"q8": "Q8", "q4": "Q4_0", "nvfp4": "NVFP4", "prune25": "prune 25%",
         "prune50": "prune 50%", "svd768": "SVD (data-free)", "svdw768": "SVD (calibrated)"}
FAMILY = {"q8": "fid", "q4": "fid", "nvfp4": "fid",
          "prune25": "fun", "prune50": "fun", "svd768": "fun", "svdw768": "fun"}
COLOR = {"q8": FID, "q4": FID2, "nvfp4": "#053061",
         "prune25": FUN, "prune50": FUN2, "svd768": "#d6604d", "svdw768": "#8c510a"}


def rate(d, arm, key):
    a = d[arm]
    return sum(a[key]) / sum(a["n"])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


lab = json.load(open(os.path.join(RES, "audit-lab.json")))
too = json.load(open(os.path.join(RES, "audit-trueood.json")))
hot = json.load(open(os.path.join(RES, "hotratio.json")))

# ---------------- F1: headline speedups ----------------
groups = ["9B short", "9B long", "27B short", "27B long"]
speed = {  # end-to-end x over plain decode, aggregate cells
    "natural": [1.52, 1.72, 2.02, 2.42],
    "trim-Q8": [1.78, 2.04, 2.15, 2.59],
    "trim-NVFP4": [1.82, 1.98, 2.15, 2.67],
    "trim-mixed": [1.81, 2.09, 2.04, 2.70],
}
fig, ax = plt.subplots(figsize=(5.0, 2.4))
w = 0.21
for i, (arm, v) in enumerate(speed.items()):
    c = [REF, FID, "#053061", "#5aae61"][i]
    x = [g + (i - 1.5) * w for g in range(4)]
    ax.bar(x, v, w, color=c, label=arm)
    for xi, vi in zip(x, v):
        ax.text(xi, vi + 0.03, f"{vi:.2f}", ha="center", fontsize=5.2, rotation=90, va="bottom")
ax.axhline(1.0, color="k", lw=0.6, ls=":")
ax.set_xticks(range(4), groups)
ax.set_ylabel("speedup over plain decode")
ax.set_ylim(0, 3.25)
ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
save(fig, "f1_headline")

# ---------------- F2: retention vs shift distance ----------------
corpora = ["in-distribution", "same-register", "domain shift"]
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for arm in ARMS:
    r = [rate(lab["indist"], arm, "acc") / rate(lab["indist"], "reference", "acc"),
         rate(lab["ood"], arm, "acc") / rate(lab["ood"], "reference", "acc"),
         rate(too["ood"], arm, "acc") / rate(too["ood"], "reference", "acc")]
    ls = "-" if FAMILY[arm] == "fid" else "--"
    ax.plot(range(3), [100 * x for x in r], ls, marker="o", ms=3,
            color=COLOR[arm], label=LABEL[arm], lw=1.4)
ax.set_xticks(range(3), corpora)
ax.set_ylabel("acceptance retention (%)")
ax.set_ylim(66, 102)
ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left")
save(fig, "f2_tax_by_family")

# ---------------- F3: per-bin damage under domain shift ----------------
bins = ["rarest", "q2", "q3", "q4", "hottest"]
fig, ax = plt.subplots(figsize=(4.2, 2.4))
show = ["nvfp4", "prune25", "svd768"]
w = 0.26
for i, arm in enumerate(show):
    ref = too["ood"]["reference"]
    a = too["ood"][arm]
    r = [(a["acc"][b] / max(ref["acc"][b], 1)) * 100 for b in range(5)]
    x = [b + (i - 1) * w for b in range(5)]
    ax.bar(x, r, w, color=COLOR[arm], label=LABEL[arm])
ax.axhline(100, color="k", lw=0.6, ls=":")
ax.set_xticks(range(5), bins)
ax.set_xlabel("token log-frequency quintile")
ax.set_ylabel("per-bin retention (%)")
ax.set_ylim(50, 108)
ax.legend(frameon=False, fontsize=7.5)
save(fig, "f3_per_bin")

# ---------------- F4: audit scatter ----------------
# tax = shift-induced retention loss: retention(in-dist) - retention(domain shift),
# the paper's definition (isolates what shift adds on top of in-dist damage)
def shift_tax(arm):
    r_in = rate(lab["indist"], arm, "acc") / rate(lab["indist"], "reference", "acc")
    r_sh = rate(too["ood"], arm, "acc") / rate(too["ood"], "reference", "acc")
    return 100 * (r_in - r_sh)

fig, ax = plt.subplots(figsize=(3.4, 2.6))
for arm in ARMS:
    dis = 1 - rate(lab["indist"], arm, "agree")
    tax = shift_tax(arm)
    m = "o" if FAMILY[arm] == "fid" else "s"
    ax.scatter(100 * dis, tax, c=COLOR[arm], marker=m, s=28, zorder=3)
    off = {"nvfp4": (5, 4), "q4": (-4, -12)}.get(arm, (5, -2))
    ax.annotate(LABEL[arm], (100 * dis, tax), textcoords="offset points",
                xytext=off, fontsize=6.5)
ax.set_xlabel("in-distribution disagreement (%)")
ax.set_ylabel("domain-shift tax (retention pts)")
ax.text(0.03, 0.94, "Spearman 1.0 (exact p = 0.0002)", transform=ax.transAxes,
        fontsize=7)
save(fig, "f4_audit_scatter")

# ---------------- F5: hot-ratio histogram ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.hist(hot["r"], bins=80, color=FID2, edgecolor="none")
ax.axvline(hot["hot_frac"], color="k", lw=1, ls="--")
ax.text(hot["hot_frac"] + 0.005, ax.get_ylim()[1] * 0.9,
        f"uniform share = {hot['hot_frac']:.3f}", fontsize=7)
ax.set_xlabel("per-channel hot-logit energy ratio $r_i$")
ax.set_ylabel("FFN channels")
ax.set_xlim(0, 0.4)
save(fig, "f5_hot_ratio")

# sanity print for the caption numbers
print("\ncaption numbers:")
for arm in ARMS:
    dis = 100 * (1 - rate(lab["indist"], arm, "agree"))
    harm_n = sum(lab["indist"]["reference"]["acc"]) - sum(lab["indist"][arm]["acc"])
    disn = sum(lab["indist"][arm]["n"]) - sum(lab["indist"][arm]["agree"])
    print(f"  {arm:8s} disagree {dis:5.2f}%  shift tax {shift_tax(arm):5.1f}pts  "
          f"harm/dis {max(harm_n,0)/max(disn,1):.3f}")
